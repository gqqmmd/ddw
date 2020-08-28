function [nor_X]=Preprocessing(data,n,m)
%min-max ��һ��������������ÿһ���㣬����ÿһ�У��������ֵ�������ֵXimax����СֵXimin,
%��׼��֮�������Xnorm=(Xi-Ximin)/(Ximax-Ximin)
%��׼��֮ǰ����ȱʡֵ������ƽ��ֵ����

%ȱʡֵ����
for i =1:n%����ÿһ����
    ind=cell(1,1);%��¼ÿһ�����ݵ��ļ�������������ȱʡ��
    %mean_value=0;%��¼��ȥȱʡֵ֮���ʣ������ֵ��ֵ��ƽ��ֵ
    for j=1:m%����ÿһ������
        if isnan(data(i,j))
            ind{1}=[ind{1},j]; 
        end
    end
    le=length(ind{1});%��¼ȱʡֵ�ĸ���
    if le%�����ȱʡֵ
        iid=ind{1}(1,:);
        data(i,iid)=0;
        mean_value=sum(data(i,:))/(m-le);
        data(i,iid)=mean_value;
    end
end






% for i =1:n
%     mean_value=zeros(n,1);%��¼��ȥȱʡֵ֮���ʣ������ֵ��ֵ��ƽ��ֵ
%     for j=1:m
%         count=cell(n,1);%��¼ÿһ�е�ȱʡֵ������
%         %ind=find(data(i,:)==NaN,2);
%         ind=find(isnan(data(i,j)));
%         if ind
%             count{j}=[count{j},ind];
%             le=length(count{j}(1,:));%���м���ȱʡֵ
%             data(ind,j)=0;
%             mean_value(i,1)=sum(data(i,:))/(n-le);
%             data(i,j)=mean_value(i,1);  
%         else
%             continue;
%         end
%     end
% end

%��С����һ��Ԥ����
nor_X=zeros(n,m);
Ximax=zeros(1,m);
Ximin=zeros(1,m);
for i=1:m
    Ximax(1,i)=max(data(:,i));
    Ximin(1,i)=min(data(:,i)); 
end
for i =1:n 
    for j=1:m
        if Ximax(1,j)~=Ximin(1,j)%��֤��ĸ��Ϊ0
             nor_X(i,j)=(data(i,j)-Ximin(1,j))/(Ximax(1,j)-Ximin(1,j));
        else
            continue;
        end
    end
end


