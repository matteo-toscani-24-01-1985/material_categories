clear all
close all

basefolder='./LMT_108_SurfaceMaterials_Database/'; %% folder from https://zeus.lmt.ei.tum.de/downloads/texture/download/LMT_108_SurfaceMaterials_Database_V1.1.zip
subfolders={'Testing/','Training/'};
folders={'AccelScansComponents/Movement/'};


%%%% READ FILENAMES
for fo=1:length(folders)
    D=[];
    for sufo=1:2
        thisfolder=[basefolder folders{fo} subfolders{sufo}];
        D=[D;  dir([thisfolder '*.txt'])];
    end
    clear ALL_NAMES Repetition
    for n=1:length(D)
        name=D(n).name;
        matname=name(3:(findstr(name,'_')-1));
        ALL_NAMES{n}=matname;
        if n<=(length(D)/2)
            ALL_FULL_NAMES{n}=[[basefolder folders{fo} subfolders{1}] name];
        else
            ALL_FULL_NAMES{n}=[[basefolder folders{fo} subfolders{2}] name];
            
        end
        try
            Repetition(n)= str2num(name((findstr(name,'test')+4):(findstr(name,'.')-1)));
        catch
            Repetition(n)= str2num(name((findstr(name,'train')+5):(findstr(name,'.')-1)));
        end
    end
    suF= round(linspace(0,1,length(D)))+1;
    uninames =unique(ALL_NAMES);
    if length(uninames)~=108;error materialsNumber;end
    
    for mat=1:length(uninames)
        for sUf=1:2
            for rep=1:10
                pos=find(contains(ALL_NAMES,uninames{mat})& (Repetition==rep) & (suF==sUf));
                if length(pos)>1
                    for p=1:length(pos)
                        if strcmp(uninames{mat} ,ALL_NAMES{pos(p)})
                            %                    ind=p;
                            pos=pos(p);
                            break
                        end
                    end
                end
                NAMES_ORDERED{rep+10*(sUf-1)+20*(mat-1),fo}=ALL_FULL_NAMES{pos};
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%
%PROCESS SAMPLES %%%
mkdir ./LMT_FORMATTED
%%%% READ  OVERALL RANGE 
RANGES=nan(length(NAMES_ORDERED),2,size(NAMES_ORDERED,2)); % measure the ranges to convert everything in the range [0 1], for the network
for i=1:length(NAMES_ORDERED)     
        fd=fopen(NAMES_ORDERED{i}); % read text files
        txt=textscan(fd,'%f');
        txt=txt{:};
        txt= txt(2128:end);      % remove 212.8 ms  - this is done also in the  "ProcessSignal.m" function: ranges are computed consistently       
        if length(txt)~=47872; error variablelength;end % check lenght
        fclose(fd);        
        RANGES(i,:)=[min(txt) max(txt)];
        disp([num2str(round(100*(i/length(NAMES_ORDERED)))) '%'])
end
Min= min(RANGES(:,1)); % compute overall min
Max= max(RANGES(:,2)); % compute overall max

% FILTER SAMPLES AND SAVE THEM %%%
% this is one with a function to use the parfor routine and save a bit of time
parfor i=1:length(NAMES_ORDERED)
    ProcessSignal(i,CATEGORIES,NAMES_ORDERED,Min,Max)
end