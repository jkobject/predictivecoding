function [Out,K,SNR]=Predict(Ka,va)

% Compute an theoritical output of the interneuron of the 1st layer of the
% retina from an input image Ka, noised by a gaussian filter of variance va
%
%
% Algorithm based on the paper from Srinivasan & al :
% Predictive coding, a fresh view of the inhibition in the retina
% http://links.jstor.org/sici?sici=0080-4649%2819821122%29216%3A1205%3C427%3APCAFVO%3E2.0.CO%3B2-P
%
%jkobject@gmail.com
% 
% July, 14, 2017
% Kalfon J?r?mie

warning('off','all')


%Ka = imread('test.jpg');
%va = 0.04;
Ka = rgb2gray(Ka);
K = imnoise(Ka,'gaussian',0,va);
psize = 10;
H = 43;
W = 64;

Patchs = zeros(2*psize+H,2*psize+W);

Patchs(psize+1:end-psize,psize+1:end-psize) = K(:,:);

Patchs(1:psize,:)= repmat(Patchs(psize+1,:),psize,1);
Patchs(end-psize+1:end,:)= repmat(Patchs(end-psize,:),psize,1);
Patchs(:,1:psize)= repmat(Patchs(:,psize+1),1,psize);
Patchs(:,end-psize+1:end)= repmat(Patchs(:,end-psize),1,psize);

Patchs = Patchs/255;
size = 2*psize;
R1 = zeros(size^2,size^2);
V = va^2;
pospix = round(((size+1)^2)/2);
Out = zeros(H-1,W-1);
Ss = zeros((H-1)*(W-1),1);
for indx = 1:H-1
    for indy = 1:W-1
        patch = Patchs(indx:indx+size,indy:indy+size);
        M = mean2(patch)^2;
        S = std2(patch);
        Ss(indx*indy) = S;
        S= S^2;
        for xIter = 1:size+1
            for yIter = 1:size+1
                for xIt = 1:size+1
                    for yIt = 1:size+1
                        if(xIt*yIt~=xIter*yIter)
                             R1(xIt*yIt, xIter*yIter) = M + S*exp((-sqrt((xIt-xIter)^2 + (yIt-yIter)^2)/psize)^2);
                        else
                             R1(xIt*yIt,xIt*yIt) =  M+S+V;
                        end
                    end
                end
            end
        end
        rhs = R1(pospix,:)';
        R = R1([1:pospix-1, pospix+1:end],:);
        
        h1 = linsolve(R', rhs);
        h = [h1(1:pospix-1); 0; h1(pospix:end)];
        
        h(h<0)=0;
        H1 = reshape(h, size+1, size+1);
        
        out = K(indx,indy) - (sum(sum(H1.*patch))*255);
        Out(indx,indy) =  out;
    end 
end
Ss = mean(Ss);
SNR = Ss/va;



        