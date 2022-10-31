# 平行程式設計-hw1
###### tags: `HW` `PP` 
# Q 1-1
> Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?
> 
測試代碼 ： ./myexp -s 10000
| Vector Width |  2 | 4 |   8   |
| -------- | -------- | -------- | -------- |
| CLAMPED EXPONENT    | 86.6%    | 81.5%  |80.9%|
ARRAY SUM |100% |100% |100%|

可以發現當Width size上升時，在CLAMPED EXPONENT時 Vector Utilization不增反降，是因為每一個element之執行長度都不同，因此會造成有些vector空置無法利用。

但在ARRAY SUM中，不管Width不管往上設多大，Vector Utilization都是不變的，這是因為每一個element所執行的長度接相同，因此Utilization皆不變。






# Q 2-2
>What speedup does the vectorized code achieve over the unvectorized code? What can you infer about the bit width of the default vector registers on the PP machines?

實驗結果
case1：make && ./test_auto_vectorize -t 1 ->nonvectorize
case2:make VECTORIZE=1 &&./test_auto_vectorize -t 1 -> nonvectorize


|          | case1    | case2    |
| -------- | -------- | -------- |
| 1.    | 8.23415sec    | 2.61615sec     |
|2.|8.26597sec |2.62138sec
|3.|8.27337sec|2.62584sec
|4.|8.24996sec|2.62158sec
|5.|8.2363sec|2.62009sec
|6.|8.24555sec|2.62269sec
|7.|8.24794sec|2.61778sec
|8.|8.26606sec|2.6263sec
|9.|8.26189sec|2.62388sec
|10.|8.28495sec|2.62957sec
|平均|8.256614sec|2.622526sec

可以看到nonvectorize與vectorize之speedup為3.148344
其中vectorize可以進行平行化處理，因此可以比nonvectorize更加快速。

且register之大小是為2的次方大小為存放空間，其中float之bit為32bit，且vectorize之sppedup為2倍，可推測同段時間有3個vector同時間在進行運算，因此可以推測register之大小 > 96bit，所以bit width of the default vector registers on the PP machines為128bit。



# Q 2-3
>Provide a theory for why the compiler is generating dramatically different assembly.

在test2中
```cpp
      /* max() */
      c[j] = a[j];
      if (b[j] > a[j])
        c[j] = b[j];
```     
這樣子寫complier會覺得會產生data dependency(RAW)，造成產生的assembly不會有平行的運算。
```cpp
+      if (b[j] > a[j]) c[j] = b[j];
+      else c[j] = a[j];
```
改成此程式碼時，就沒有RAW的問題，因此comlier就會進行平行化處理。