function [output] = ColorConstancy(input)
dimension=size(input,3);
input=im2uint8(input);
output=zeros(size(input));    
if (dimension==1 || dimension==3)
for j=1:dimension
Value=sum(sum(input(:,:,j)))/numel(input(:,:,j));
output(:,:,j)=input(:,:,j)*(127.5/Value);
end
output=uint8(output);
else 
error('Input error');
end