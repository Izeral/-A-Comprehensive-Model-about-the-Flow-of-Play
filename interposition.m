clc
clear
data1=xlsread('插补结果.xlsx');
data1(:,9)=data1(:,6);
data1(:,6)=[];
data=data1;
for i=size(data,1):-1:1
    if data(i,5)==-1
        data(i,:)=[];
    end
end
for i=size(data,1):-1:1
    if data(i,6)==-1
        data(i,:)=[];
    end
end
for i=size(data,1):-1:1
    if data(i,7)==-1
        data(i,:)=[];
    end
end
for i=size(data,1):-1:1
    if data(i,8)==-1
        data(i,:)=[];
    end
end
newdata=[];
for i=1:size(data,1)
    newdata(i,1)=data(i,data(i,1));
end
data(:,1)=newdata;
data(:,2:3)=[];

%% BP神经网络
train_ratio=0.8;%用作训练集的比例
train1=data(1:(floor(train_ratio*size(data, 1))),:);%前80%用于训练
x_train=train1(:,1:end-1);
y_train=train1(:,end);
test1=data(((floor(train_ratio*size(data, 1)))+1):size(data, 1),:);%后20%用于测试
x_test=test1(:,1:end-1);
y_test=test1(:,end);
% 进行标准化
[x_train_normalized, mu, sigma] = zscore(x_train);
x_test_normalized = (x_test - mu) ./ sigma;


%建立网络
net=feedforwardnet([10 10 10]);%建立一个隐含层数为1，节点数为10的网络;这里用了梯度下降的训练函数

%定义在训练集过程中不使用测试集，只保留训练集与验证集（用于验证泛化能力）
net.divideParam.trainRatio = 80/100; %训练集
net.divideParam.valRatio = 20/100; %验证集
net.divideParam.testRatio = 0/100; %测试集

% net.trainParam.epochs=5000;%指定最大迭代次数
% net.trainParam.goal=0.4;%目标精度，验证集的误差达到该数字时就停止迭代
% net.trainParam.min_grad=1e-2;%最小下降梯度
% net.trainParam.lr=0.1;%学习率
net.trainparam.max_fail =30;%持续无法优化次数（泛化能力）
% net.layers{2}.transferFcn= 'logsig';%指定传递函数
%训练网络。
net=train(net,x_train_normalized',y_train');%注意，这里每一列为一个实例，所以需要转置
%测试效果
test_out=sim(net,x_test_normalized');%测试
test_out=test_out';
%分类
test_out=round(test_out);
% 画图
color=[0/255,96/255,115/255;
    9/255,147/255,150/255;
    145/255,211/255,192/255;
    235/255,215/255,165/255;
    238/255,155/255,0/255;
    204/255,102/255,2/255;
    188/255,62/255,3/255;
    174/255,32/255,18/255;
    155/255,34/255,39/255];
scatter([1:size(y_test,1)],y_test)
hold on
plot([1:size(y_test,1)],test_out,'color',color(1,:))
plot([1:size(y_test,1)]',ones(size(y_test,1),1)*mean(y_train),'LineWidth', 2,'color','r')
legend('真实值','预测值','根据均值插补')
set(gcf,'Color',[1 1 1])

%% 开始插补
newdata=[];
for i=1:size(data1,1)
    newdata(i,1)=data1(i,data1(i,1));
end
data1(:,1)=newdata;
data1(:,2:3)=[];
x_need=[];
for i=1:size(data1,1)
   if data1(i,6)==-1
       x_need=[x_need;data1(i,1:5)];
   end
end
if ~isempty(x_need)
    x_need_normalized = (x_need - mu) ./ sigma;
    need_out=sim(net,x_need_normalized');%测试
    need_out=need_out';
    need_out=round(need_out);
    num=0;
    for i=1:size(data1,1)
        if data1(i,6)==-1
            num=num+1;
            data1(i,6)=need_out(num);
        end
    end
end