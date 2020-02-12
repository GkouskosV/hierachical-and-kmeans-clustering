function [idx, clusters, cluster_centroids] = extract_cluster_info(Data, cl_num, solution)
    
% Store the num of dimentions
dim = size(Data, 2);

cluster_centroids = zeros(cl_num, dim);

% Initialize 2 1*cl_num cell arrays 
idx = cell(1, cl_num);
clusters = cell(1, cl_num);

for k = 1:1:cl_num  
    % Store the location of each point in the corresponding k-th cell array
    idx{k} = find(solution==k);
    % Store the actual data points in the clusters cell array
    clusters{k} = Data(idx{k},:);
    % Find each cluster's mean
    cluster_centroids(k,:) = mean(clusters{k});
end
    