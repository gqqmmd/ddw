function [f1,auc]=Direct(D,num)
  %% �˹����ݼ� %%%%%%%%%%%%%%%%%%%%%%%%%
  X=D(:,1:2);
  Y=D(:,3);
  %% UCI���ݼ� %%%%%%%%%%%%%%%%%%%%%%%%%
  %D=HTRU_2 
%   X1=D(:,1:8);
%   Y=D(:,9);

%D=shuttle1
% X1=D(:,1:9);
% Y=D(:,10);

% D=risk_factors_cervical_cancer;
%  X1=D(:,1:35);
%  Y=D(:,36);
 
 %D=Breast cancer Wisconsin
%  X1=D(:,1:10);
%  Y=D(:,11);

%D=zoo %ADD������COF����õ�
% X1=D(:,1:17);
% Y=D(:,18);

% %D=winequality_white2 %ADD0.2ʱ������COF�����ǵ�0.1ʱ���
% X1=D(:,1:11);
% Y=D(:,12);

% D=BreastCancerCoimbra;
%  X1=D(:,1:9);
%  Y=D(:,10);

% % %D=ionosphere;
%  X1=D(:,1:34);
%  Y=D(:,35);

 %D=processdcle;
%  X1=D(:,1:13);
%  Y=D(:,14);

%D=transfusion;
% X1=D(:,1:4);
% Y=D(:,5);

 % %D=audit_risk;
%  X1=D(:,1:26);
%  Y=D(:,27);
 
 % % %D=iris;  
%  X1=D(:,1:4);
%  Y=D(:,5);
 
 % % %D=iris2; 
%  X1=D(:,1:4);
%  Y=D(:,5);


% %D=SPECTFNew;
%  X1=D(:,1:44);
%  Y=D(:,45);

% % D=seeds
%  X1=D(:,1:7);
%  Y=D(:,8);

% % D=banknote
%  X1=D(:,1:4);
%  Y=D(:,5);

% % D=hepatitis
%  X1=D(:,1:19);
%  Y=D(:,20);

% % D=Lymphography
%  X1=D(:,2:19);
%  Y=D(:,1);

% % D=heart
%  X1=D(:,1:13);
%  Y=D(:,14);

% % D=wine
%  X1=D(:,2:14);
%  Y=D(:,1);

%%  �������� %%%%%%%%%%%%
%   [n,m]=size(X1); %���ݵĸ���n
%   [X]=Preprocessing(X1,n,m);
  
  [n,m]=size(X); %���ݵĸ���n
  dist=pdist2(X, X); 
  [Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n);
  %����ÿ���㣬����������ÿ����Ȼ�ھӵĵ�λ����
  Unit_Direct=cell(n,1);
  for i=1:n
      for j=1:Neicount
          w =NaN{i}(1,j);
          x= X(w,1) - X(i,1);
          y= X(w,1) - X(i,2);
          v=sqrt(power(x,2)+power(y,2));
          Unit_Direct{i}(j,1)=x/v;
          Unit_Direct{i}(j,2)=y/v;
      end
          Unit_Direct{i}(i,1)=0;
          Unit_Direct{i}(i,2)=0;
  end
  %������Ȩ�ؾ���
  %����ÿ���㣬�������ھ�����������ÿ�����������������ĵ�����ҳ��������0.5�Ľ�����ͣ�
  %�����Ǹ������������ոõ��Ŀ�귽�򣬴�ʱ�ҳ��͸õ�ĵ������0.5�ĵ���������ֱ����Щ����м�Ȩ
  %ÿ������һ���������Ŀ�����ô��Ȩ�ؾͼ�1
 Target_Vec=zeros(n,2);
 Start_count=zeros(n,1);
 point=zeros(n,1);
 point_weight=zeros(n,1);
  for i=1:n
      sum_dj=zeros(Neicount,1);
      for j=1:Neicount
          for k =1:Neicount
                dj = Unit_Direct{i}(j,1)*Unit_Direct{i}(k,1)+ Unit_Direct{i}(j,2)*Unit_Direct{i}(k,2);
                if dj>0.5
                     sum_dj(j,1)=sum_dj(j,1)+dj;
                end
          end
      end
      [i_max_sum,index]=max(sum_dj); 
      Target_Vec(i,1)= Unit_Direct{i}(index,1);
      Target_Vec(i,2)= Unit_Direct{i}(index,2);
      for g=1:Neicount
           dj_temp = Unit_Direct{i}(index,1)*Unit_Direct{i}(g,1)+ Unit_Direct{i}(index,2)*Unit_Direct{i}(g,2);
           if dj_temp>0.5
                point(NaN{i}(1,g),1)=point(NaN{i}(1,g),1)+1;
                Start_count(i,1)= Start_count(i,1)+1;
           end
      end
  end
 for i =1:n
      point_weight(i,1)=point(i,1)+Start_count(i,1);
      %point_weight(i,1)=point(i,1);
  end
 %������Ȩ�ؾ���
 %����һ���ǶԳ�Ȩ�ؾ���ÿ���ߵ�Ȩ�ؾ������ݶ������以�������ݶ����Ϊ�˴˵��ھ������������������
 %[edge_weight]=findWeight(dist,n,Sup);
 edge_weight=zeros(n,n);
 %����ÿһ���㣬�������Ļ���������NN�����е�λ�ü�Ϊ������������
for i=1:n
    for j=1:Neicount
        k1=find(NN{i}==NaN{i}(1,j));
        k2=find(NN{NaN{i}(1,j)}==i);
        if isempty(k1)
            k1=0;
        end
        if isempty(k2) 
            k2=0;
        end
        edge_weight(i,NaN{i}(1,j))=max(k1,k2);
    end
end


%  for i=1:n
%      for j=1:Neicount
%           mm=min(edge(i,NaN{i}(1,j)),edge(NaN{i}(1,j),i));
%           edge_weight(i,NaN{i}(1,j))=mm;
%           edge_weight(NaN{i}(1,j),i)=mm;
%      end
% end

 %����ÿ����ľֲ��ܶ�
 density=zeros(n,1);
 avg_point_weight=zeros(n,1);
 avg_edge_weight=zeros(n,1);
 avg_dist=zeros(n,1);
 for i=1:n
     sum_point_weight=0;
     sum_edge_weight=0;
     sum_dist=0;
     for j=1:Neicount
         sum_point_weight=sum_point_weight+point_weight(NaN{i}(1,j),1);
         sum_edge_weight=sum_edge_weight+edge_weight(i,NaN{i}(1,j));
         sum_dist=sum_dist+dist(i,NaN{i}(1,j));
     end
     avg_point_weight(i,1)=sum_point_weight/Neicount;
     avg_edge_weight(i,1)=sum_edge_weight/Neicount;
     avg_dist(i,1)=sum_dist/Neicount;
 end

%��ÿ�����������Membership
MS=zeros(n,1);
for i=1:n
    if  Neicount
        MS(i,1) = point_weight(i,1)/avg_dist(i,1);
    end
end 

%����ÿ����Ĵ�����
 density=zeros(n,1);
 for i=1:n
     if Neicount
          density(i,1)= MS(i,1)/avg_edge_weight(i,1);
     end
 end
label=zeros(n,1);
sort_density=sort(density,'ascend');
ff=sort_density(num,1);%ȡ����num��С��ֵ
 for i =1:n
     if density(i,1)<=ff
         label(i,1)=1;
     end
 end
 
%  [acc,rp]=Evaluating(n,label,Y);
 
 
%�����������ݼ�
scatter(X(label==0,1),X(label==0,2),40,'g','.');hold on 
scatter(X(label==1,1),X(label==1,2),40,'k','.');hold on
scatter(X(label==1,1),X(label==1,2),40,[0.5,0.5,0.5],'*');
outliers=find(label==1);%�쳣�������
Scores = zeros(n,1);
Scores(outliers) = 1; % outlier scores
proclass = 1; 
[~,~,~,auc] = perfcurve(Y,Scores,proclass);

%����F1ֵ
f1= F1(Y,label);
end


