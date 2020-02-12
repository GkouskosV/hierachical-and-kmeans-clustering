% This script demonstrates the hierarchical clustering algorithm on a set of
% artificially generated data points. Data points are considered to be
% sampled from three different multivariate distributions identified by the
% parameters MU1, MU2, MU3 and SIGMA1, SIGMA2, SIGMA3.

%Initialize workspace.
clc
clear all

% Define the parameters of the multivariate gaussian distributions.
MU1 = [4 4];
MU2 = [6 6];
MU3 = [8 8];
SIGMA1 = [1 0; 0 1];
SIGMA2 = [1 0; 0 1];
SIGMA3 = [1 0; 0 1];

% Sample equal number of points from both distributions.
% Let N be the number of points to be sampled.
N = 200;
X1 = mvnrnd(MU1,SIGMA1,N);
X2 = mvnrnd(MU2,SIGMA2,N);
X3 = mvnrnd(MU3,SIGMA3,N);
% Store both sets of points in a single matrix.
X = [X1;X2;X3];

% Plot the labeded data points
figure('Name','Labeled Data Popints')
hold on
plot(X1(:,1),X1(:,2),'*r','LineWidth',1.4);
plot(X2(:,1),X2(:,2),'*b','LineWidth',1.4);
plot(X3(:,1),X3(:,2),'*g','LineWidth',1.4);
xlabel('x1');
ylabel('x2');
grid on
hold off

% Plot the unlabaled data points.
figure('Name','Unlabeled Data Points')
hold on
plot(X(:,1),X(:,2),'*k','LineWidth',1.4);
xlabel('x1');
ylabel('x2');
grid on
hold off

%% Code Segment 1.
% Plot the Probability Density Function for the pair-wise distances between 
% points within the complete dataset. 

% Find euclidean distances for every point in the dataset
% and store them in a Matrix
Distance = dist(X, X');
Right = size(X,1);

% Save the lowest, highest and the mean distance found in the dataset
min_dist = min(min(Distance));
max_dist = max(max(Distance));
mean_dist = mean(reshape(Distance, 1, numel(Distance)));

% Find the space between the lowest and highest distance and the space
% inerval
S = max_dist - min_dist;
n = 50;
dx = S / n;


f = zeros(1,n);
% find all the zeros in the matrix and store treir location in an 1D array
v_zeros = find(Distance==0);
f_zeros = length(v_zeros);
%{
if(f_zeros > Right)
   f_zeros = f_zeros - Right;
end;
%}

for k = 1:1:n
    
    if (k == 1)
        % Find all points that are greater than 0 and lower than the space
        % interval and store them in an array
        vf = find(and((Distance>(k-1)*dx),(Distance<k*dx)));
        test_length_vf = length(vf);
    elseif (k == n)
        vf = find(and((Distance>=(k-1)*dx),(Distance<=k*dx)));
    else
        % find all points that are greater or equal than the space interval
        % populated by each k-1 && the points that lower than the enterval
        % populated by k
        vf = find(and((Distance>=(k-1)*dx),(Distance<k*dx)));
    end

    % Add the length of every k_th array of points in the f_array 
    % and divide by 2 because the entries are doubled
    f(k) = length(vf)/2;
end

% Add the zeros to the first bar
f(1) = f(1) + f_zeros;
x = [1:1:n];
% Enter the center of every bar
x = x.*dx;

% Create the frequences
sum_f = sum(f);
f = f / sum_f;

title_str = strcat(['D_{min} = ' num2str(min_dist) ' D_{max} = ' num2str(max_dist) ' D_{mean} = ' num2str(mean_dist)]);
figure('Name', 'PDF of the dinsances between the data points');
hold on
bar(x,f,'red')
grid on
xlabel('Distance');
ylabel('Frequecy');
title(title_str);
grid minor
hold off



%% Code Segment 2.
% Experiment 1 Perform k-means clustering by setting the number of clusters 
% to 2.
num_of_clusters = 2;
%Store the cluster solution and the centers in the 2dim space
[cl_solution, centroids] = kmeans(X, num_of_clusters, 'distance', 'sqeuclidean', 'Replicates', 5);

[idx, clusters, cluster_centroids] = extract_cluster_info(X, num_of_clusters, cl_solution);
plot_clusters(X, idx);



%% Experiment 2 Perform k-means clustering by setting the number of clusters 
% to 3.

num_of_clusters = 3;
[cl_solution, centroids] = kmeans(X, num_of_clusters, 'distance', 'sqeuclidean', 'Replicates', 5);

[idx, clusters, cluster_centroids] = extract_cluster_info(X, num_of_clusters, cl_solution);
plot_clusters(X, idx);




%% Experiment 3 Perform k-means clustering by setting the number of clusters 
% to 7.

num_of_clusters = 7;
[cl_solution, centroids] = kmeans(X, num_of_clusters, 'distance', 'sqeuclidean', 'Replicates', 5);

[idx, clusters, cluster_centroids] = extract_cluster_info(X, num_of_clusters, cl_solution);
plot_clusters(X, idx);


%% Code Segment 3.
% Create the hierarchical clustering dendrogram.

Y = pdist(X, 'euclidean');
Z = linkage(Y, 'average');
figure('Name', 'Hierarchical clustering dendogram')
H = dendrogram(Z, 10, 'Orientation', "top", 'colorthreshold', "default");
set(H, 'linewidth', 2);


%% Code Segment 4.
% Cluster data in three clusters.
T = clusterdata(X,'Linkage','ward','Maxclust', 3);
figure('Name','Identified Clusters')
scatter(X(:,1),X(:,2),10,T,'filled')
grid on