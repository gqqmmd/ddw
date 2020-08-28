function [f1,auc]=Direct(D,num)
  %% 人工数据集 %%%%%%%%%%%%%%%%%%%%%%%%%
  X=D(:,1:2);
  Y=D(:,3);
  %% UCI数据集 %%%%%%%%%%%%%%%%%%%%%%%%%
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

%D=zoo %ADD仅次于COF是最好的
% X1=D(:,1:17);
% Y=D(:,18);

% %D=winequality_white2 %ADD0.2时仅次于COF，但是当0.1时最好
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

%%  处理数据 %%%%%%%%%%%%
%   [n,m]=size(X1); %数据的个数n
%   [X]=Preprocessing(X1,n,m);
  
  [n,m]=size(X); %数据的个数n
  dist=pdist2(X, X); 
  [Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n);
  %对于每个点，求它到它的每个自然邻居的单位向量
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
  %建立点权重矩阵
  %对于每个点，求它的邻居向量集合中每个向量和其他向量的点积，找出点积大于0.5的进行求和，
  %最大的那个向量就是最终该点的目标方向，此时找出和该点的点积大于0.5的点的索引，分别给这些点进行加权
  %每被当做一次其他点的目标点那么点权重就加1
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
 %建立边权重矩阵
 %建立一个非对称权重矩阵，每条边的权重就是数据对象与其互近邻数据对象成为彼此的邻居所需的最少搜索次数
 %[edge_weight]=findWeight(dist,n,Sup);
 edge_weight=zeros(n,n);
 %对于每一个点，返回他的互近邻在其NN集合中的位置即为最少搜索次数
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

 %计算每个点的局部密度
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

%求每个点的隶属度Membership
MS=zeros(n,1);
for i=1:n
    if  Neicount
        MS(i,1) = point_weight(i,1)/avg_dist(i,1);
    end
end 

%计算每个点的从属力
 density=zeros(n,1);
 for i=1:n
     if Neicount
          density(i,1)= MS(i,1)/avg_edge_weight(i,1);
     end
 end
label=zeros(n,1);
sort_density=sort(density,'ascend');
ff=sort_density(num,1);%取到第num个小的值
 for i =1:n
     if density(i,1)<=ff
         label(i,1)=1;
     end
 end
 
%  [acc,rp]=Evaluating(n,label,Y);
 
 
%画出最后的数据集
scatter(X(label==0,1),X(label==0,2),40,'g','.');hold on 
scatter(X(label==1,1),X(label==1,2),40,'k','.');hold on
scatter(X(label==1,1),X(label==1,2),40,[0.5,0.5,0.5],'*');
outliers=find(label==1);%异常点的索引
Scores = zeros(n,1);
Scores(outliers) = 1; % outlier scores
proclass = 1; 
[~,~,~,auc] = perfcurve(Y,Scores,proclass);

%计算F1值
f1= F1(Y,label);
end


