clear;
close all;
NumOfTrainingSamples = 20; 
NumOfEigenface = 10;
training_Path = '/Users/apple/Documents/MATLAB/pca/PCA_Images/Training/';

fprintf(1,'Loading Training Images from ',training_Path,' ... ');
for i = 1: NumOfTrainingSamples
    str_Load = strcat(training_Path, num2str(i), '.bmp'); 
    Image = imread(str_Load);
    TrainingImage(:,i) = double(reshape(Image, [ ], 1));
end
fprintf(1,'Done\n');

[ImageHeight,ImageWidth] = size(Image);
[row,col] = size(TrainingImage);

fprintf(1,'Calculating Mean Face ... ');
MeanFace = zeros(row,1);
for i = 1: NumOfTrainingSamples
    MeanFace = MeanFace+TrainingImage(:,i);
end
MeanFace = MeanFace/NumOfTrainingSamples;
fprintf(1,'Done\n');

fprintf(1,'Showing Mean Face ... ');
figure;
imagesc(reshape(MeanFace, [ImageHeight ImageWidth]))
title('Average Face','FontSize', 18);
colorbar
colormap(gray)
fprintf(1,'Done\n');

fprintf(1,'Calculating Demeaned Face ... ');
for i = 1: NumOfTrainingSamples
    DemeanFace(:,i) = TrainingImage(:,i) - MeanFace;
end
fprintf(1,'Done\n');

fprintf(1,'Showing Demeaned Face ... ');
figure;
for i = 1: NumOfTrainingSamples
    Display = DemeanFace(:,i);
    Display = reshape(Display, [ImageHeight ImageWidth]);
    subplot(NumOfTrainingSamples/5,5,i);
    imagesc(Display) 
    colorbar
    colormap(gray)
end
axes('Units','Normal');
t = title('Demeaned Face','FontSize', 18);
set(gca,'visible','off')
set(t,'visible','on')
set(gcf, 'Position', [100 100 1240 840])
fprintf(1,'Done\n');

fprintf(1,'Calculating Covariance Matrix and its Eigenvalues and Eigenvectors ... ');
CovFace1 = cov(DemeanFace');
[EV, ED] = eig(CovFace1);
for i = 1 : length(ED)
    Eigenvalue(i) = ED(i,i);
end
Eigenvalue_sorted = sort(Eigenvalue,'descend');
for i = 1 : length(ED)
    for j = 1 : length(ED)
        if Eigenvalue_sorted(i) == Eigenvalue(j)
            order(i) = j;
        end
    end
end
fprintf(1,'Done\n');

fprintf(1,'Reconstructing faces from %i Eigenfaces ... ',NumOfEigenface);
ReconstImage = zeros(row,NumOfTrainingSamples);
figure;
for i = 1: NumOfTrainingSamples
    for j = 1:NumOfEigenface % N is the number of eigenvectors available 
        coef(i,j) = DemeanFace(:,i)'*EV(:,order(j));
        ReconstImage(:,i) = ReconstImage(:,i)+coef(i,j)*EV(:,order(j));
    end
    ReconstImage(:,i) = MeanFace + ReconstImage(:,i);
    subplot(NumOfTrainingSamples/5,5,i);
    imagesc(reshape(ReconstImage(:,i), [ImageHeight ImageWidth]))
    colorbar
    colormap(gray)
end
axes('Units','Normal');
t = title('Recontructed Face','FontSize', 18);
set(gca,'visible','off')
set(t,'visible','on')
set(gcf, 'Position', [100 100 1240 840])
fprintf(1,'Done\n');

fprintf(1,'Cauclating the SSE between original faces and reconstructed faces ... ');
for i = 1: NumOfTrainingSamples
    Difference(:,i) = TrainingImage(:,i) - ReconstImage(:,i);
    SSE(i) = sum(sum(Difference(:,i).*Difference(:,i)));
end
figure;
plot(SSE);
title('Error Sum of Squares (SSE)')
fprintf(1,'Done\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testng
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumOfTestingSamples = 4;

testing_Path = '/Users/apple/Desktop/FYP/PCA/PCA_Images/Testing/';
fprintf(1,'Loading Testing Images from ',testing_Path,' ... ');
for i = 1: NumOfTestingSamples
    str_Load = strcat(training_Path, num2str(i), '.bmp'); 
    Image = imread(str_Load);
    TestingImage(:,i) = double(reshape(Image, [ ], 1));
end
fprintf(1,'Done\n');

fprintf(1,'Calculating Demeaned Testing Face ... ');
for i = 1: NumOfTestingSamples
    DemeanTestingFace(:,i) = TestingImage(:,i) - MeanFace;
end
fprintf(1,'Done\n');

fprintf(1,'Showing Demeaned Testing Face ... ');
figure;
for i = 1: NumOfTestingSamples
    Display = DemeanTestingFace(:,i);
    Display = reshape(Display, [ImageHeight ImageWidth]);
    subplot(1,NumOfTestingSamples,i);
    imagesc(Display) 
    colorbar
    colormap(gray)
end
axes('Units','Normal');
t = title('Demeaned Testing Face','FontSize', 18);
set(gca,'visible','off')
set(t,'visible','on')
set(gcf, 'Position', [100 100 1240 840])
fprintf(1,'Done\n');

fprintf(1,'Reconstructing testing faces from %i Eigenfaces ... ',4608);
ReconstTestingImage = zeros(row,NumOfTestingSamples);
figure;
for i = 1: NumOfTestingSamples
    for j = 1:4608 % N is the number of eigenvectors available 
        coef(i,j) = DemeanTestingFace(:,i)'*EV(:,order(j));
        ReconstTestingImage(:,i) = ReconstTestingImage(:,i)+coef(i,j)*EV(:,order(j));
    end
    ReconstTestingImage(:,i) = MeanFace + ReconstTestingImage(:,i);
    subplot(1,NumOfTestingSamples,i);
    imagesc(reshape(ReconstTestingImage(:,i), [ImageHeight ImageWidth]))
    colorbar
    colormap(gray)
end
axes('Units','Normal');
t = title('Recontructed TestingFace','FontSize', 18);
set(gca,'visible','off')
set(t,'visible','on')
set(gcf, 'Position', [100 100 1240 840])
fprintf(1,'Done\n');

fprintf(1,'Cauclating the SSE between original faces and reconstructed testing faces ... ');
for i = 1: NumOfTestingSamples
    TestingDifference(:,i) = TestingImage(:,i) - ReconstTestingImage(:,i);
    TestingSSE(i) = sum(sum(TestingDifference(:,i).*TestingDifference(:,i)));
end
figure;
plot(TestingSSE);
title('Error Sum of Squares (SSE) for testing faces')
fprintf(1,'Done\n');
