% clear everything and start fresh
clear;
clc;

% seed the random number generator to get a repeatable output 
rng(3);

% import the data from a csv file
data = readtable('data.csv');

% correct the header names since MATLAB does not accept "(" and ")" in the
% variable names
data = renamevars(data, ...
    ["No_", "Length_m_", "Height_m_", "Deflection_cm_", "Failure", "FailureProbability"],...
    ["no", "length_m", "height_m","deflection_cm", "failure", "failure_probability"]);

% split the table into dependent and independent variables
inputs = data(:, ["length_m", "height_m"]);
deflection = data(:, "deflection_cm");
failure = data(:, "failure");

% convert the table into matrix
inputs = table2array(inputs);
deflection = table2array(deflection);
failure = table2array(failure);

% convert the failure classes (yes/no) into one-hot encoded vectors to be
% used in the patternnet (nprtool)
failure = categorical(failure);
failure_classes = categories(failure);
failure = onehotencode(failure, 2);


% create and train the feed forward ANN with 10 hidden nodes and default
% parameters for deflection prediction
deflection_model = feedforwardnet(10);
deflection_model = train(deflection_model, transpose(inputs), transpose(deflection));

% create and train the pattern recognition ANN with 10 hidden nodes and default
% parameters for failure prediction
failure_model = patternnet(10);
failure_model = train(failure_model, transpose(inputs), transpose(failure));

% use the trained network to predict the given cases
given_inputs = [48, 3.2;
                87, 7.8];
predicted_deflection = deflection_model(transpose(given_inputs));

predicted_failure = failure_model(transpose(given_inputs));
predicted_failure = vec2ind(predicted_failure);
predicted_failure = failure_classes(predicted_failure);

results = table([41;42],predicted_deflection', predicted_failure, 'VariableNames', ["No.", "Predicted deflection (cm)", "Predicted failure"]);
results
