initialTheta = [1; 2];
options = optimset('GradObj', 'on', 'MaxIter', 100);
[optTheta, functionVal, exitFlag] = ...
    fminunc(@costFunction, initialTheta, options);