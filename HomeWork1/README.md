[**Description**](https://docs.google.com/presentation/d/1r2u-xVytctdRSbaCAHwWlHIBkmJ50Stnpj1hqi9pFXs/edit?usp=sharing)   


## Predicting PM2.5  

## Data   
- train.csv:每个月前20天每个小时的气象资料（每小时有18种测资）共12个月
- test.csv: 排除train.csv剩余的资料 取连续9小时的资料当作feature 预测第10小时的PM2.5值 总共240个
- ans.csv test的answer

## Feature Selection
**热力图分析**：由热力图可直接看出，与PM2.5相关性较高的指标有PM10、NO2、SO2、NOX、O3、THC.

![](https://github.com/Everglowww/ML2017_Hung-yi-Lee_HomeWork/blob/master/HomeWork1/figures/FetureSlection.png) 
选择最具代表性的特征：**PM2.5**、**PM10**、**SO2**




## Result
使用SGD以及Adagrad的方式对数据进行20000次的训练  
得出来的w会与close-form的值进行对比

![](https://github.com/Everglowww/ML2017_Hung-yi-Lee_HomeWork/blob/master/HomeWork1/figures/TrainProcess.png)  
Adagrad在很快的次数内就很接近close-form的解   
SGD在20000次内并不能够得到最优解，但是速度很快   


![](https://github.com/Everglowww/ML2017_Hung-yi-Lee_HomeWork/blob/master/HomeWork1/figures/Compare.png)   
图上红色的小点表示test data的正确值    
由此我们可以看到adagrad以及close-form的预测值已经很接近真实值  
由于训练次数不够SGD与真实值还有一些差距  
虽然代码内部有加L2 regularization，但是由于模型简单，并没有什么明显改善
