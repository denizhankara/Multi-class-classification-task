trainData.csv içerisindeki 103 adet öznitelik kullanılarak (v1,v2,...v103,  target kolonu hedef etiket olarak kullanılmak üzere) makina öğrenmesi algoritmaları ve teknikleri kullanılarak 
multi-class logloss metriği minimize edilecektir. Bu metriğin hesaplanması için klasör içerisindeki metric_MC_logloss.jpeg resmindeki formulden yararlanabilirsiniz, bazı ML algorithmaları 
gömülü olarak bu metriği barındırabilir, ya da bazı kütüphanelerde hazır olarak bulunabilir.

Veri seti oldukça temiz ve rahatlıkla birçok ML tekniği denenebilir. Beklentimiz feature engineering, ML modelling,  ensembling etc. yöntemlerinin verimli kullanıldığını görmek.
İyi sonuç almak için state-of-the-art yöntemler ve algoritmaları kullanmanızı tavsiye ederiz. 
 

İyi Çalışmalar,



Some notes

- Calibrated model neg log loss scores (cross validated)
[-0.57120038 -0.5474214  -0.55356751 -0.53711963 -0.57223535 -0.54380967
 -0.56156468 -0.56440363 -0.55612542 -0.54642795 -0.56879571 -0.57003458
 -0.55010096 -0.57022679 -0.54670483 -0.53154089 -0.58108508 -0.54116473
 -0.54113452 -0.54543001 -0.55236497 -0.56174692 -0.55473804 -0.55384316
 -0.57670146 -0.5435231  -0.546548   -0.54710394 -0.57347231 -0.54751134]


Dry run on selected features
[-0.49217221 -0.47493618 -0.47525532 -0.45667162 -0.49499099 -0.46648747
 -0.47918846 -0.4788286  -0.49100557 -0.46870498]
Mean ROC AUC: -0.478

1) Embed a classifer to differentiate classes 2-3:

Accuracy: 82.06%
Log_loss: 0.480075
[[ 218    9    1    3    1   31   13   53   57]
 [   1 2748  381   36    4   11   25    5   13]
 [   0  672  856   20    0    7   42    3    1]
 [   0  161   75  285    0    9    7    0    1]
 [   1    8    1    0  536    1    1    0    0]
 [  11   19    6    4    0 2683   24   45   35]
 [   8   48   46    5    3   34  387   30    7]
 [  25   12    3    0    0   38   15 1579   21]
 [  38   14    0    1    1   25    3   45  864]]


Accuracy: 86.82%
Log_loss: 0.308570
[[7189  243  119]
 [ 148 2708  368]
 [  94  659  848]]



Accuracy: 82.72%
Log_loss: 0.763323
[[ 218    9    1    3    1   31   13   53   57]
 [   1 2760  371   35    4   11   24    5   13]
 [   0  614  915   23    0    7   37    3    2]
 [   0  157   75  288    0   10    7    0    1]
 [   1    8    1    0  536    1    1    0    0]
 [  12   16    6    4    0 2690   24   42   33]
 [   8   46   47    6    4   34  386   30    7]
 [  26   12    2    0    0   38   13 1581   21]
 [  38   14    0    1    1   25    3   45  864]]



 2) Train a general classifer, then for labels 2-3-4,
 use first classifier probs to get a finer result as final
 decision--