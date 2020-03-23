clear all

load('matlab_labeling2.mat')

for e = 1:507
    msk = zeros(1024,1536);
    [m,n] = size(msks.blacke{e});
    b = msks.blacke{e};
    for o = 1:m
        %rectangle: 
        %msk(b(o,2):b(o,2)+b(o,4),b(o,1):b(o,1)+b(o,3)) = 1;
        %elipse:
        cX = b(o,1)+b(o,3)/2;
        cY = b(o,2)+b(o,4)/2;
        rX = b(o,3)/10;
        rY = b(o,4)/10;
        [columnsInImage, rowsInImage] = meshgrid(1:1536, 1:1024);
        ellipsePixels = (rowsInImage - cY).^2 ./ rY^2 + (columnsInImage  - cX).^2 ./ rX^2 <= 1;
        msk =  msk + ellipsePixels;
    end
    msk = msk>0;
    [filepath,name,ext] = fileparts(msks.imageFilename{e});
    
    %path = strcat('D:/10_GitHub/Agroscope/data/00_all/masks_matlab4/',name,'.png');
    path = strcat('C:/Users/dschori/Documents/01_git/Agroscope/data/00_all/masks_matlab4/',name,'.png');
    imwrite(msk,path);
    e
end