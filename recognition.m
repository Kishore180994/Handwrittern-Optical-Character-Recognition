%Image Preprocessing
I = imread('test1.png');
final_image = 'test1.png';
%Convert to GrayScale
Igray =rgb2gray(I);
 imshow(I)
pause;
%Convert to bi  nary image
Ibw = im2bw(Igray,graythresh(Igray));
% imshow(Ibw);
% pause;
%Edge detection
Iedge = edge(uint8(Ibw));
% imshow(Iedge);
% pause;
% Image Dilation
se = strel('square',2);
Iedge2 = imdilate(Iedge, se);
% imshow(Iedge2);
% pause;
%Image Filling
Ifill= imfill(Iedge2,'holes');
% imshow(Ifill);
% pause;
%Blob analysis
[Ilabel num] = bwlabel(Ifill);
disp(num);
Iprops = regionprops(Ilabel);
Ibox = [Iprops.BoundingBox];
Ibox = reshape(Ibox,[4 num]);
 imshow(I)

%Plot the Object Location
% hold on;
for cnt = 1:num
    rectangle('position',Ibox(:,cnt),'edgecolor','r');
end
% for cnt1 = 1:num
%     data=imcrop(Ifill,Ibox(:,cnt1));
%     
%     %Feature Extraction
%     %Trimming Image
%     data( ~any(data,2), : ) = [];  %rows
%     data( :, ~any(data,1) ) = [];  %columns
%     %imshow(data)
%     %resizing Image
%     final_data = imresize(data,[5 7]);
%     data1(cnt1,:)=reshape(final_data',1,35); %data to be sent to neural network
% end
cnt=1;
data1=[];
row1=[];
row2=[];
row3=[];
row4=[];
row5=[];
p2=[];
for cnt1 = 1:num
    data=imcrop(Ifill,Ibox(:,cnt1));
    
    %Feature Extraction
    %Trimming Image
    data( ~any(data,2), : ) = [];  %rows
    data( :, ~any(data,1) ) = [];  %columns
    %resizing Image
    final_data = imresize(data,[5 7]);
    %data to be sent to neural network
    data1=[data1 reshape(final_data',35,1)];
    if(rem(cnt1,5)==1)
        row1=[row1 reshape(final_data',35,1)];
    end
    if(rem(cnt1,5)==2)
        row2=[row2 reshape(final_data',35,1)];
    end
    if(rem(cnt1,5)==3)
        row3=[row3 reshape(final_data',35,1)];
    end
    if(rem(cnt1,5)==4)
        row4=[row4 reshape(final_data',35,1)];
    end
    if(rem(cnt1,5)==0)
        row5=[row5 reshape(final_data',35,1)];
    end
        
end
out=[row1 row2 row3 row4 row5];
% out21=data1(1:4,:);
% out22=data1(6:9,:);
% out23=data1(11:14,:);
% out24=data1(16:19,:);
% out11=[data1(21:24,:);data1(26:29,:);data1(31:34,:);data1(36:39,:);data1(41:44,:);data1(46:49,:)];
% out25=[data1(5,:);data1(10,:);data1(15,:);data1(20,:);data1(25,:);data1(30,:);data1(35,:);data1(40,:);data1(45,:);data1(50,:)];
% out=[out21;out22;out23;out24;out11;out25];
%Training with the help of training function 
% net = edu_createnn(P,T); 
P = out(:,1:40); 
T = [eye(10) eye(10) eye(10) eye(10)]; 
Ptest = out(:,43); 
%BackPropagation Training
[w1,w2,b1,b2] = bpa(P,35,40,30,10,T);

%Testing with feed forward
answer = ffn(w1,w2,b1,b2,Ptest,35,1,30,10);

final = char(answer.*[49 50 51 52 53 'A' 'B' 'C' 'D' 'E'])


% net = newff(P,T,[35], {'logsig'}) 
% %net.performFcn = 'sse'; 
% net.divideParam.trainRatio = 1; % training set [%]
% net.divideParam.valRatio   = 0; % validation set [%] 
% net.divideParam.testRatio  = 0; % test set [%] 
% net.trainParam.goal = 0.001; 
% [net,tr,Y,E] = train(net,P,T);
% 
% %% Testing the Neural Network 
% [a,b]=max(sim(net,Ptest)); 
% disp(b); 




