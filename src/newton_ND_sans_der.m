function [approx , err_abs] = newton_ND_sans_der(F , x0 , nb_it_max , tol_rel , file_name)
% NEWTON_ND_SANS_DER	Methode de Newton pour la resolution de F(x) = 0, pour F: R^n -> R^n
%
% Syntaxe: [approx , err_abs] = newton_ND_sans_der(F , x0 , nb_it_max , tol_rel , file_name)
%
% Arguments d'entree
%	F			-	String ou fonction handle specifiant la fonction
%					non-lineaire (F: R^n -> R^n)
%	x0			-	Approximation initiale (x0 dans R^n)
%	nb_it_max	-	Nombre maximum d'iterations 
%	tol_rel		-	Tolerance sur l'approximation de l'erreur relative
%	file_name	-	(Optionnel) Nom du fichier (avec l'extension .txt) dans
%					lequel sera	ecrit les resultats de l'algorithme
%
% Arguments de sortie
%	approx		-	Matrice de taille (n x nb_iter) contenant les 
%					iterations
%	err_abs		-	Vecteur rangee de dimension nb_iter contenant les
%					erreurs absolues
%
% Exemples d'appel
%	[ approx , err_abs ] = newton_ND_sans_der( 'my_sys_nl' , [1;1] , 20 , 1e-9 , 'resul_newtonND.txt')
%	[ approx , err_abs ] = newton_ND_sans_der( @(x) [x(1)^2 + x(2)^2 - 1 ; -x(1)^2 + x(2)] , [1;1] , 20 , 1e-9 , 'resul_newtonND.txt')



%%  Verification de la fonction F
if isa(F,'char')
	fct		=	str2func(F);
	is_fct_file =	true;
elseif isa(F,'function_handle')
	fct		=	F;
	is_fct_file =	false;
else
	error('L''argument f n''est pas un string ni un function_handle')
end


%% Verification du nb de composantes des approximations initiales et de F
if ~isnumeric(x0) || ~isvector(x0)
	error('L''approximation initiale x0 n''est pas un vecteur')
end

x0_col	=	reshape(x0,[],1);

try 
	fct(x0_col);
catch ME
	rethrow(ME)
end

taille	=	length(x0_col);

if ~isnumeric(fct(x0_col)) || ~isvector(fct(x0_col)) || (length(fct(x0_col))~=taille)
	error(['Le fonction F ne retourne pas un vecteur de meme taille que x0. ',...
		'x0 est de taille %d alors que F(x0) est de taille %d.'],taille,length(fct(x0_col)))
end

%% Verification du fichier output
if nargin == 5 && ~isa(file_name,'char')
	error('Le nom du fichier des resultats doit etre de type string')
end

%% Initialisation des matrices app et err
app			=	nan(taille,nb_it_max);
app(:,1)	=	x0_col;
err_rel		=	inf(1,nb_it_max);
arret		=	false;


%% Methode de Newton
for t=1:nb_it_max-1
	
	app_jac		=	app_jacobienne(fct,app(:,t));
	delta_x		=	app_jac\-reshape(fct(app(:,t)),taille,1);
	app(:,t+1)	=	app(:,t) + delta_x;
	
	if any(~isfinite(app_jac(:)))
		warning(['La matrice jacobienne de f a l''iteration %d est singuliere 0.\n',...
					'Arret de l''algorithme'],t)
		break
	end
	
	if min(abs(eig(app_jac))) == 0  
		warning(['La matrice jacobienne de f a l''iteration %d est singuliere 0.\n',...
					'Arret de l''algorithme'],t)
		break
	end
	
	err_rel(t)	=	norm(app(:,t+1)-app(:,t))/(norm(abs(app(:,t+1))) + eps);
	if (err_rel(t) <= tol_rel) || (norm(fct(app(:,t+1))) == 0)
		arret	=	true;
		break
	end
	
end

nb_it	=	t+1;
approx	=	app(:,1:nb_it);
err_abs	=	inf(1,nb_it);

if arret
	for t=1:nb_it
		err_abs(t)	=	norm(approx(:,end) - approx(:,t));
	end
else
	warning('La methode de Newton n''a pas convergee')
end

% ecriture des resultats si fichier passe en argument
if nargin == 5
	output_results(file_name , fct , is_fct_file , ...
				nb_it_max , tol_rel , x0 , approx , err_abs , arret)
end

end


function [app_finale] = app_jacobienne(f,x0)

	taille	=	length(x0);
	if min(x0) == 0
		h_init	=	1e-6;
	else
		h_init	=	1e-3 * min(x0);
	end
	h		=	h_init./(2.^(0:1));
	app		=	cell(2,1);
	
	for t=1:length(h)
		app{t}			=	zeros(taille,taille);
		for d=1:taille
			delta_h		=	zeros(size(x0));
			delta_h(d)	=	h(t);
			app{t}(:,d)	=	(f(x0+delta_h) - f(x0-delta_h))/(2*h(t));
		end
	end
	
	app_finale	=	(2^2*app{2} - app{1})/(2^2-1);
end

function [] = output_results(file_name , fct , is_fct_file , ...
			      it_max , tol_rel , x0 , x , err , status)
						 
	fid		=	fopen(file_name,'w');
	fprintf(fid,'Algorithme de Newton approximant la matrice jacobienne\n\n');
	fprintf(fid,'Fonction dont on cherche les racines:\n');
	if is_fct_file
		fprintf(fid,'%s\n\n',fileread([func2str(fct),'.m']));
	else
		fprintf(fid,'%s\n\n',func2str(fct));
	end

	fprintf(fid,'Arguments d''entree:\n');
	fprintf(fid,'    - Nombre maximum d''iterations: %d\n',it_max);
	fprintf(fid,'    - Tolerance relative: %6.5e\n',tol_rel);
	fprintf(fid,'    - Approximation initiale x0: [');
	[taille, nb_iter]	=	size(x);
	if taille <= 3
		fprintf(fid,'%6.5e ',x0(1:taille));
		fprintf(fid,']\n');
	else
		fprintf(fid,'%6.5e ',x0(1:2));
		fprintf(fid,'... %6.5e]',x0(end));
	end
	
	if status
		fprintf(fid,'\nStatut: L''algorithme de Newton a converge en %d iterations\n\n',nb_iter-1);
	else
		fprintf(fid,'\nStatut: L''algorithme de Newton n''a pas converge\n\n');
	end
	if taille == 1
		fprintf(fid,'#It       x_1           Erreur absolue\n');
		fprintf(fid,'%3d   %16.15e   %6.5e\n',[reshape(0:nb_iter-1,1,[]);reshape(x(:),1,[]);reshape(err,1,[])]);
	elseif taille == 2
		fprintf(fid,'#It       x_1           x_2       Erreur absolue\n');
		fprintf(fid,'%3d   %6.5e   %6.5e   %6.5e\n',[reshape(0:nb_iter-1,1,[]);reshape(x([1,2],:),2,[]);reshape(err,1,[])]);
	elseif taille == 3
		fprintf(fid,'#It       x_1           x_2           x_3       Erreur absolue\n');
		fprintf(fid,'%3d   %6.5e   %6.5e   %6.5e   %6.5e\n',[reshape(0:nb_iter-1,1,[]);reshape(x([1,2,3],:),3,[]);reshape(err,1,[])]);
	else
		fprintf(fid,'#It       x_1           x_2      ...       x_n       Erreur absolue\n');
		fprintf(fid,'%3d   %6.5e   %6.5e  ...  %6.5e   %6.5e\n',[reshape(0:nb_iter-1,1,[]);reshape(x([1,2,taille],:),3,[]);reshape(err,1,[])]);
	end
	
	fclose(fid);

end
