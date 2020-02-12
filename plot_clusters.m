function plot_clusters(Data,cluster_indices)

% This function generates a 3-dimensional plot for the cluster distribution
% of a given dataset.


% Set all possible line specifiers in order to indicate distinct clusters.
MarkerSpecifiers = ['*','o','+','x','s','d'];
ColorSpecifiers = ['r','g','b','c','m','y','k'];
MarkerSpecifiersNum = length(MarkerSpecifiers);
ColorSpecifiersNum = length(ColorSpecifiers);
LineSpecifiersNum = MarkerSpecifiersNum * ColorSpecifiersNum;
LineSpecifiers = cell(1,LineSpecifiersNum);
line_specifier_index = 1;
for marker_specifier_index = 1:1:MarkerSpecifiersNum
    for color_specifier_index = 1:1:ColorSpecifiersNum
        LineSpecifiers{line_specifier_index} = strcat(MarkerSpecifiers(marker_specifier_index),ColorSpecifiers(color_specifier_index));
        line_specifier_index = line_specifier_index + 1;
    end;
end;


% Retrieve the number of clusters.
ClustersNum = length(cluster_indices);
clusters = cell(1,ClustersNum);
for k = 1:1:ClustersNum
    clusters{k} = Data(cluster_indices{k},:);
end;

% Create new figure.
figure('Name','Cluster Distribution');
plot(clusters{1}(:,1),clusters{1}(:,2),LineSpecifiers{1});
hold on
for k = 2:1:ClustersNum
    % The internal cluster (k) index must be mapped to an integer within 
    % the range [1..LineSpecifiersNum]. This integer is the
    % line_specifier_index which will be obtained as:
    % line_specifier_index = (k mod LineSpecifiersNum) + 1 ensuring that
    % its value will be strictly within the specified interval.
    line_specifier_index = mod(k,LineSpecifiersNum) + 1;
    plot(clusters{k}(:,1),clusters{k}(:,2),LineSpecifiers{line_specifier_index});
end;
grid on
hold off

end