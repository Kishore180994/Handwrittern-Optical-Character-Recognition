b1=load('bias1.mat');
b2=load('bias2.mat');
w1=load('weights1.mat');
w2=load('weights2.mat');
b1=b1.b1;
b2=b2.b2;
w1=w1.w1;
w2=w2.w2;
%Image Preprocessing
I = imread('test1.png');
final_image = 'test1.png';
%Convert to GrayScale
Igray =rgb2gray(I);
% imshow(Igray);
% pause;
%Convert to binary image
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
imshow(Ifill);
% pause;
%Blob analysis
[Ilabel num] = bwlabel(Ifill);
disp(num);
Iprops = regionprops(Ilabel);
Ibox = [Iprops.BoundingBox];
Ibox = reshape(Ibox,[4 num]);
hold on;
for cnt = 1:num
    rectangle('position',Ibox(:,cnt),'edgecolor','r');
end
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
j=1;
for i=41:50
Ptest = out(:,i); 

answer = ffn(w1,w2,b1,b2,Ptest,35,1,30,10);

final(j,1) = char(answer*[49;50;51;52;53;'A';'B';'C';'D';'E']);
j=j+1;
end
final