function [temps , y] = euler_exp(f , tspan , Y0 , nb_pas)
% EULER_EXP	Methode d'Euler explicite avec pas constant pour la
%			resolution d'EDOs
%
% Arguments d'entree
%	f		-	String ou function handle definissant le systeme de N EDOs
%	tspan	-	Vecteur contenant le temps initial et final [t0,tf]
%	x0		-	Vecteur colonne contenant les N conditions initiales
%	nb_pas	-	Nombre de pas de temps
%
% Arguments de sortie
%	temps	-	Vecteur rangee contenant les valeurs de temps t_i
%	y		-	Matrice de dimension N x (nb_pas+1) dont les rangees 
%				sont les approximations de y_i(t)
%
% Exemples d'appel
%	[temps , y] = euler_exp('my_edo' , [0,1] , [1;0] , 1000);
%	[temps , y] = euler_exp(@(t,y) y*cos(t) , [0,2] , 1 , 1000 );
%	[temps , y] = euler_exp(@(t,z) [z(2);-10*z(1)] , [0,1] , [1;0] , 1000);



%%  Verification de la fonction contenant la definition des derivees
if isa(f,'char')
	fct		=	str2func(f);
elseif isa(f,'function_handle')
	fct		=	f;
else
	error('L''argument f n''est pas un string ni un function_handle')
end

%% Verification du temps et nb pas de temps
if ~isnumeric(tspan) || length(tspan)~=2
	error('Le vecteur tspan doit contenir 2 composantes, [t0 , tf]')
elseif ~isscalar(nb_pas) || floor(nb_pas)~=nb_pas ...
						  || length(nb_pas)~=1 || nb_pas<0
	error('Le nombre de pas nb_pas doit etre entier et positif')
end
t0			=	tspan(1);
tf			=	tspan(2);
nb_pas		=	double(nb_pas);
h			=	(tf-t0)/nb_pas;

%% Verification du nb de composantes des conditions initiales et de f
if ~isnumeric(Y0) || ~isvector(Y0)
	error('Les conditions initiales x0 ne sont pas arrangees en vecteur')
end

Y0_col = reshape(Y0,[],1);

try 
	fct(t0,Y0_col);
catch ME
	rethrow(ME)
end

if ~isnumeric(fct(t0,Y0_col)) || ~isvector(fct(t0,Y0_col))
	error('La f ne retourne pas un vecteur')
elseif length(Y0_col) ~= length(fct(t0,Y0_col))
	error('Le nombre de composantes de x0 et f ne concorde pas')
end

nb_comp		=	length(Y0_col);

%% Initialisation du temps et de la matrice y
temps		=	reshape(linspace(t0,tf,nb_pas+1),1,nb_pas+1);
y			=	nan(length(Y0_col), nb_pas + 1);
y(:,1)		=	Y0_col;

%% Methode d'Euler explicite avec pas constant
for t=1:nb_pas
	yt			=	y(:,t);
	y(:,t+1)	=	yt + h*reshape(fct(temps(t),yt),nb_comp,1);
end

end