function ProcessSignal(i,CATEGORIES,NAMES_ORDERED,Mins,Maxs)

tmp=[];
if exist(['./LMT_FORMATTED/C' num2str(CATEGORIES(i)) 'SAMP' num2str(i) '.txt'])==0
    for dime=1:size(NAMES_ORDERED,2)
        fd=fopen(NAMES_ORDERED{i,dime});
        txt=textscan(fd,'%f');
        txt=txt{:};
        txt= txt(2128:end);
        % filter with matlab routines
        txt= highpass(txt,10,10000) ;
        %%%% LINEAR TRANSFORM --- to force the range between zero and one.
        txt=( txt-Mins(dime))/(Maxs(dime)-Mins(dime));
        fclose(fd);
        tmp=[tmp;txt'];
    end
      writematrix(tmp,['./LMT_FORMATTED/SAMP' num2str(i) '.txt'],'Delimiter',' ')
end
disp([num2str(round(100*(i/length(NAMES_ORDERED)))) '%'])