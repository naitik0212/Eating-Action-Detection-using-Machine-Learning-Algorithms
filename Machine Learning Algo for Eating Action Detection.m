Eat = importdata('Eat.csv');
NonEat = importdata('NonEat.csv');
% phase 1
accuracy_Mat_dtree = zeros(33,6);
accuracy_Mat_svm = zeros(33,6);
accuracy_Mat_neural = zeros(33,6);

for i = 1:33 
    eat_user = Eat(((i-1)*78)+1:(i*78),:);
    noneat_user = NonEat(((i-1)*76)+1:(i*76),:);
    
    data = [eat_user; noneat_user];
    data = data(randperm(end),:);
    train_data = data(1:floor(size(data)*0.6),:);
    test_data = data((floor(size(data)*0.6)+1):size(data),:);
    
    for j = 1:3
        if j==1
        % decision tree
            model = fitctree(train_data(:,1:18),train_data(:,19));
            predicted_labels = predict(model,test_data(:,1:18));
            
        elseif j==2
        % svm    
             model = fitcsvm(train_data(:,1:18),train_data(:,19));
             predicted_labels = predict(model,test_data(:,1:18));
        else
        % neural net
             train_data = transpose(train_data);
             t = double(train_data(19,:));
             t = [t; 1-t];
             net = patternnet(32);
             [net, model] = train(net, train_data(1:18,:), t);
             test_data1 = transpose(test_data);
             predicted_labels = net(test_data1(1:18,:));
             predicted_labels = predicted_labels > 0.5;
             predicted_labels = predicted_labels(1,:);
             predicted_labels = double(transpose(predicted_labels));
        end
        
        

        confusion = confusionmat(test_data(:,19),predicted_labels);

        precision = confusion(1,1) ./ (confusion(1,1)+confusion(2,1));
        recall = confusion(1,1) ./ (confusion(1,1)+confusion(1,2));

        F1 = (2*recall*precision) ./ (recall+precision);

        TPR = recall;
        FPR = confusion(2,1) ./ (confusion(2,1)+confusion(2,2));
        
        [X,Y,T,AUC] = perfcurve(test_data(:,19),predicted_labels,1);
       
        if j==1
            accuracy_Mat_dtree(i,:) = [precision recall F1 TPR FPR AUC];
        elseif j==2
            accuracy_Mat_svm(i,:) = [precision recall F1 TPR FPR AUC];
        else
            accuracy_Mat_neural(i,:) = [precision recall F1 TPR FPR AUC];
        end
        if(i==33)
            title('ROC CURVE')
            xlabel('False Positive Rate (FPR)') % x-axis label
            ylabel('True Positive Rate (TPR)') % y-axis label
            plot(X,Y)
            hold on
        end
    end
    saveas(gcf, 'SINGLE_USER', 'jpg');
    cla reset;
end

%output phase1

a=[1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33];
C = horzcat(a, accuracy_Mat_dtree);

task1Mat = [C accuracy_Mat_svm accuracy_Mat_neural];

modelHeader = {' ','Decision Tree','Decision Tree','Decision Tree','Decision Tree','Decision Tree','Decision Tree','SVM','SVM','SVM','SVM','SVM','SVM','Neural Networks','Neural Networks','Neural Networks','Neural Networks','Neural Networks','Neural Networks',};
row = {'User','Precision','Recall','F1Score','TPR','FPR','RoC AUC','Precision','Recall','F1Score','TPR','FPR','RoC AUC','Precision','Recall','F1Score','TPR','FPR','RoC AUC',};
fid = fopen('Results_Phase1.csv', 'w') ;

fprintf(fid, '%s,', modelHeader{1,1:end-1}) ;

fprintf(fid, '%s\n', modelHeader{1,end}) ;

fprintf(fid, '%s,', row{1,1:end-1}) ;

fprintf(fid, '%s\n', row{1,end}) ;

fclose(fid) ;

dlmwrite('Results_Phase1.csv', task1Mat, '-append') ;


% phase 2
accuracy_Mat_dtree_full = zeros(23,6);
accuracy_Mat_svm_full = zeros(23,6);
accuracy_Mat_neural_full = zeros(23,6);

train_data = [Eat(1:780,:); NonEat(1:760,:)];
train_data = train_data(randperm(end),:);


model1 = fitctree(train_data(:,1:18),train_data(:,19));

model2 = fitcsvm(train_data(:,1:18),train_data(:,19));


train_data = transpose(train_data);
t = double(train_data(19,:));
t = [t; 1-t];
net1 = patternnet(32);
[net1, model3] = train(net1, train_data(1:18,:), t);

for i = 11:33
    eat_user = Eat(((i-1)*78)+1:(i*78),:);
    noneat_user = NonEat(((i-1)*76)+1:(i*76),:);
    test_data = [eat_user; noneat_user];
    test_data = test_data(randperm(end),:);
    
    for j = 1:3
        if j==1
        % decision tree
            
            predicted_labels = predict(model1,test_data(:,1:18));
        elseif j==2
        % svm    
             
             predicted_labels = predict(model2,test_data(:,1:18));
        else
        % neural net
             
             test_data1 = transpose(test_data);
             predicted_labels = net1(test_data1(1:18,:));
             predicted_labels = predicted_labels > 0.5;
             predicted_labels = predicted_labels(1,:);
             predicted_labels = double(transpose(predicted_labels));
        end
        
        

        confusion = confusionmat(test_data(:,19),predicted_labels);

        precision = confusion(1,1) ./ (confusion(1,1)+confusion(2,1));
        recall = confusion(1,1) ./ (confusion(1,1)+confusion(1,2));

        F1 = (2*recall*precision) ./ (recall+precision);

        TPR = recall;
        FPR = confusion(2,1) ./ (confusion(2,1)+confusion(2,2));
        
        [X,Y,T,AUC] = perfcurve(test_data(:,19),predicted_labels,1);
        
        if j==1
            accuracy_Mat_dtree_full(i-10,:) = [precision recall F1 TPR FPR AUC];
        elseif j==2
            accuracy_Mat_svm_full(i-10,:) = [precision recall F1 TPR FPR AUC];
        else
            accuracy_Mat_neural_full(i-10,:) = [precision recall F1 TPR FPR AUC];
        end
        if(i==33)
            title('ROC CURVE')
            xlabel('False Positive Rate (FPR)') % x-axis label
            ylabel('True Positive Rate (TPR)') % y-axis label
            plot(X,Y)
            hold on
        end
    end
            saveas(gcf, 'ALL_USERS', 'jpg');
end

%output phase2
b=[1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23];
D = horzcat(b, accuracy_Mat_dtree_full);


task2Mat = [D accuracy_Mat_svm_full accuracy_Mat_neural_full];

modelHeader = {' ','Decision Tree','Decision Tree','Decision Tree','Decision Tree','Decision Tree','Decision Tree','SVM','SVM','SVM','SVM','SVM','SVM','Neural Networks','Neural Networks','Neural Networks','Neural Networks','Neural Networks','Neural Networks',};
row = {'User','Precision','Recall','F1Score','TPR','FPR','RoC AUC','Precision','Recall','F1Score','TPR','FPR','RoC AUC','Precision','Recall','F1Score','TPR','FPR','RoC AUC',};
fid = fopen('Results_Phase2.csv', 'w') ;

fprintf(fid, '%s,', modelHeader{1,1:end-1}) ;

fprintf(fid, '%s\n', modelHeader{1,end}) ;

fprintf(fid, '%s,', row{1,1:end-1}) ;

fprintf(fid, '%s\n', row{1,end}) ;

fclose(fid) ;

dlmwrite('Results_Phase2.csv', task2Mat, '-append') ;

