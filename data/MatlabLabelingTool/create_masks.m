clear all

load('matlab_labeling.mat')

for e = 1:507
    msk = zeros(1024,1536);
    [m,n] = size(msks.blacke{e});
    b = msks.blacke{e};
    for o = 1:m
        msk(b(o,2):b(o,2)+b(o,4),b(o,1):b(o,1)+b(o,3)) = 1;
    end
    [filepath,name,ext] = fileparts(msks.imageFilename{e});
    path = strcat('D:/10_GitHub/Agroscope/data/00_all/masks_matlab/',name,'.png');
    imwrite(msk,path);
    e
end

imshow(msk)