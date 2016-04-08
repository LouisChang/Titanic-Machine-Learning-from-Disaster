function [ err ] = bagclassif( XTRAIN,YTRAIN,XTEST,YTEST )
%ENSEMBLE OF BAGGED DECISION TREES CLASSIFICATION Summary of this function
% goes here
%   Detailed explanation goes here
bag = fitensemble(XTRAIN,YTRAIN,'Bag',500,'Tree',...
    'type','classification');
predtest = bag.predict(XTEST);
%err = sum(~arrayfun(@ismember,predtest,YTEST))/length(YTEST);
%err = sum(~arrayfun(@ismember,predtest,YTEST))
err = sum(~strcmp(predtest,YTEST))
end

