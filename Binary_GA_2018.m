function Feat_Index =  Binary_GA_2018
                    
clear all
global Data 
% tez_data.mat Verisini yükle  
% Verideki X (Feature) ve Y (Target)

Data  = load('tez_data.mat'); 
Column =33; % Datasetteki Sütun Sayýsý
tournamentSize = 2;
options = gaoptimset('CreationFcn', {@PopFunction},...
                     'PopulationSize',50,...
                     'Generations',100,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectiontournament,tournamentSize},...
                     'MutationFcn',{@mutationuniform, 0.1},...
                     'CrossoverFcn', {@crossoverarithmetic,0.8},...
                     'EliteCount',2,...
                     'StallGenLimit',100,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rand('seed',1)
nVars = 33; % 
FitnessFcn = @FitFunc_KNN; 
[chromosome,~,~,~,~,~] = ga(FitnessFcn,nVars,options);
Best_chromosome = chromosome; % En iyi Kromozom
Feat_Index = find(Best_chromosome==1); % En iyi kromozomun indexi
end

%%% POPULATION FUNCTION
function [pop] = PopFunction(Column,~,options)
RD = rand;  
pop = (rand(options.PopulationSize, Column)> RD); % Popülasyon
end

%%% FITNESS FUNCTION   
function [FitVal] = FitFunc_KNN(pop)
global Data
FeatIndex = find(pop==1); %Feature Index
X1 = Data.X;% Features Set
Y1 = grp2idx(Data.Y);% Target bilgisi
X1 = X1(:,[FeatIndex]);
NumFeat = numel(FeatIndex);
Compute = ClassificationKNN.fit(X1,Y1,'NSMethod','exhaustive','Distance','euclidean'); 
Compute.NumNeighbors = 5; % kNN = 5
FitVal = resubLoss(Compute)/(34-NumFeat);
end

