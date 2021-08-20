clc
clear

import matlab.unittest.TestSuite;
suite = TestSuite.fromFolder(pwd);
result = run(suite);