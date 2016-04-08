function [ err ] = dtreeclassif( XTRAIN,YTRAIN,XTEST,YTEST )
%DECISION TREE CLASSIFICATION Summary of this function goes here
%   Detailed explanation goes here
tree = ClassificationTree.fit(XTRAIN,YTRAIN);
predtest = tree.predict(XTEST);
%err = sum(~arrayfun(@ismember,predtest,YTEST));
err = sum(~strcmp(predtest,YTEST));
end

