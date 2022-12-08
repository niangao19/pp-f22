# 平行程式作業-hw4
###### tags:  `HW` `PP`
# PART1
## Q1
>1-1 How do you control the number of MPI processes on each node?
:::info
在hostfile裡面的hostname後面加上slots=N，代表以該hostname為node僅能執行N個processes
:::

```hostfile=
pp2 slots=2
pp3 slots=1
```

> 1-2 Which functions do you use for retrieving the rank of an MPI process and the total number of processes?
:::info
Retrieving the rank:
:::
```cpp=
int rank;
MPI_Comm_rank( MPI_COMM_WORLD, &rank);
```
:::info
Retrieving total number of processes:
:::
```cpp=
int size;
MPI_Comm_size( MPI_COMM_WORLD, &size);
```


## Q2

>2-1 Why MPI_Send and MPI_Recv are called “blocking” communication? 

:::info
使用 MPI_Send後，須等到有人接收訊息才會繼續往下執行，MPI_Recv也是要等到要接收的訊息送出後，並且接收到訊息才會繼續往下執行，因此稱為 “blocking” communication
:::

> 2-2 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.


| np-num   | 1        | 2        |4        |8        |16       |
| -------- | -------- | -------- |-------- |-------- |-------- |
| time(sec)| 13.441949|    7.019677  | 3.425933| 1.742128|0.986151


![block_linear](https://i.imgur.com/WVlglXk.png)


## Q3
> 3-1 Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.
>

| np-num   | 1        | 2        |4        |8        |16       |
| -------- | -------- | -------- |-------- |-------- |-------- |
| time(sec)| 13.52962| 6.969916 | 3.418616| 1.736754|0.90741|

![](https://i.imgur.com/rScOqxF.png)
> 3-2How does the performance of binary tree reduction compare to the performance of linear reduction? 


![](https://i.imgur.com/YTCBGDj.png)

:::info
將前面兩個測試結果，畫成圖表，可以觀察到這兩個時間差別並不多，因此在目前來看，兩者的效率差不多。
但以第三題的點後解釋來看linear得效率比較好。
:::



> 3-3 Increasing the number of processes, which approach (linear/tree) is going to perform better? Why?
> 
| np-num    | 2       | 4        | 8       |16        |
| -------- | -------- | -------- |-------- | -------- | 
| linear-加法次數     | 1     | 3     |     7 |        15|
| linear-通訊次數     | 1     | 3     |      7|        15|
| tree-加法次數(rank0)   | 1       | 2     |     3|       4 |
| tree-通訊次數 (rank0)  | 1       | 2     |     3|        4 |
| tree-加法次數(all)   | 1       | 3     |     7|        15|
| tree-通訊次數 (all)  | 1       | 3     |     7|        15|
:::info
以上面的圖表來看tree的rank0的通訊還有加法次數比linear還要少，但是以總體來看通訊次數以及加法次數是一樣的，但是tree會花額外的成本來計算要發送哪一個process或接收哪一個process，因此linear會比tree效率還要高。
:::

## Q4
> 4-1 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.
 
| np-num   | 1        | 2        |4        |8        |16       |
| -------- | -------- | -------- |-------- |-------- |-------- |
| time(sec)| 13.303059| 6.73126 | 3.502156| 1.718273|0.867591

![](https://i.imgur.com/8rxVsPM.png)



> 4-2 What are the MPI functions for non-blocking communication? 
:::info
* MPI_Isend
* MPI_Irecv
* MPI_Wait
* MPI_Waitall
:::

> 4-3 How the performance of non-blocking communication compares to the performance of blocking communication?
:::info
non-blocking communication的performance會比較佳，因為每一個的processes運算時間都不一樣，blocking的需要等前面的確認有接收到後才能去接受下一個，而non-blocking的則是可以先跟所有的processes去做通訊連接，因此速度上non-blocking的速度會比較快
:::
## Q5
> 5-1 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| np-num   | 1        | 2        |4        |8        |16       |
| -------- | -------- | -------- |-------- |-------- |-------- |
| time(sec)| 13.599173| 6.690827 |3.409834| 1.723295|0.8831

![](https://i.imgur.com/zRZOYGK.png)

## Q6
> 6-1 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| np-num   | 1        | 2        |4        |8        |16       |
| -------- | -------- | -------- |-------- |-------- |-------- |
| time(sec)| 12.902159|    6.649385  | 3.412531| 1.731311|0.872791

![](https://i.imgur.com/mCQ94Gv.png)




# PART2
## Q7
> Describe what approach(es) were used in your MPI matrix multiplication for each data set.

* 從rank0讀入matrixA與matrixB以及他們的大小，並使用MPI_Bcast和MPI_Scatter傳送訊息。

* 將B matrix transpose，因為這樣子才不會導致cache 做page replacement，因為在乘法的時候B是取column，但是電腦是以row major來做存放的因此改成row的形式存放，取資料的時候已row的形式取出，速度會比較快(約2倍)


```cpp=
    for( int i = 0; i < local_n; i++ ) {
        for ( int k = 0; k < l; k++)  {
            result_local[ i * l + k] = 0;
            bn = k*m;
            for (  int j = 0 ; j < m; j++ ) {
                result_local[i* l + k] += a_mat[i*m+j]* b_mat[bn];
                bn++;
            }
        }
    }
```
* 每一個processes進行矩陣乘法後，在使用MPI_Gather將結果取回到rank0主機。

參考資料:[MPI介紹](https://chenhh.gitbooks.io/parallel_processing/content/mpi/mpich_p2p_prog.html)