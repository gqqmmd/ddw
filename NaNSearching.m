function[Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n)
%D=varargin{1};
r=1;
nb=zeros(n,1);
C=cell(n,1);
NN=cell(n,1);%��ʼ��ÿ�����KNN�ھ�
RNN=cell(n,1);%��ʼ��ÿ�����RKNN�ھ�
%NNN=cell(n,1);%��NN��RNN�Ľ�����Ҳ��ÿ�������Ȼ�ڸ���
NaN=cell(n,1);
%A=pdist2(D,D);
Numb1=0;
Numb2=0;
for ii=1:n
   [sa,index]=sort(dist(:,ii));
   C{ii}=[sa,index];
end
while(r<n)
    for kk=1:n
        x=kk;
        y=C{x}(r+1,2);%�ҵ���Ӧ������
        nb(y)= nb(y)+1;
        NN{x}=[NN{x},y];
        RNN{y}=[RNN{y},x];
    end
    Numb1=sum(nb==0);
    if Numb2~=Numb1
        Numb2=Numb1;
    else
       break; 
    end
    r=r+1;
end
for jj=1:n
    NNN{jj}=intersect(NN{jj},RNN{jj});
end
Neicount=r;
for jj =1:n
    temp=C{jj}(2:r+1,2);
    NaN{jj}=temp';
end
end