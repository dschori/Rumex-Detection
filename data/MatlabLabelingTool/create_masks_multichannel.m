clear all

load('matlab_labeling2.mat')

radius = 40;

counter = 0;

for e = 1:507
    msk_ellipsis = zeros(1024,1536);
    msk_circle = zeros(1024,1536);
    msk_multi = zeros(1024,1536,3);
    [m,n] = size(msks.blacke{e});
    b = msks.blacke{e};
    for o = 1:m
        %rectangle: 
        %msk(b(o,2):b(o,2)+b(o,4),b(o,1):b(o,1)+b(o,3)) = 1;
        %elipse:
        cX = b(o,1)+b(o,3)/2;
        cY = b(o,2)+b(o,4)/2;
        rX = b(o,3)/2.5;
        rY = b(o,4)/2.5;
        rX_small = b(o,3)/6.2;
        rY_small = b(o,4)/6.2;
        [columnsInImage, rowsInImage] = meshgrid(1:1536, 1:1024);
        ellipsePixels = (rowsInImage - cY).^2 ./ rY^2 + (columnsInImage  - cX).^2 ./ rX^2 <= 1;
        circlePixels = (rowsInImage - cY).^2 ./ rY_small^2 + (columnsInImage  - cX).^2 ./ rX_small^2 <= 1;
        msk_ellipsis =  msk_ellipsis + ellipsePixels;
        msk_circle =  msk_circle + circlePixels;
    end
    if max(msk_circle(:)) > 1
        counter = counter + 1
    end
    msk_ellipsis = msk_ellipsis>0;
    msk_circle = msk_circle>0;
    msk_multi(:,:,1) = msk_ellipsis;
    msk_multi(:,:,2) = msk_circle;
    [filepath,name,ext] = fileparts(msks.imageFilename{e});
    
    path = strcat('D:/10_GitHub/Agroscope/data/00_all/masks_matlab4/',name,'.png');
    %path = strcat('C:/Users/dschori/Documents/01_git/Agroscope/data/00_all/masks_matlab2/',name,'.png');
    imwrite(msk_multi,path);
    %e
end