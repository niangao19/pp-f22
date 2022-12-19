# 平行程設HW5
###### tags:  `HW` `PP`

## Q1
>What are the pros and cons of the three methods? Give an assumption about their performances.
![](https://i.imgur.com/czXRBsM.png)

### kernel1
  - pros:
    - 每一個thread只有一個pixel處理
    - 因為memory 是Pageable的，所以performance不會下降。
  - cons: 
    - 快的thread要等慢的thread
    - 若需要的page不再memory裡需要等待傳輸
![](https://i.imgur.com/UwaSmiS.png)

### kernel2
  - pros:
    - 每一個thread只有一個pixel處理
    - 因為所需的memory是pinned的所以會比較快。
    - cudaMallocPitch使用padding的方式對齊，使每個row開頭照256或512的倍數對齊，提高訪問效率。
  - cons: 
    - 快的thread要等慢的thread
    - padding會浪費記憶體的空間
    - cudaHostAlloc如果Alloc太多的話會降低performance因為pinned memory無法paging
### kernel3
  - pros:
    - 同kernel2 
    - 可以解決計算量不均的問題(剛好計算量重的配上輕的)
    -  一個thread可以處理多個pixel，如果GPU有限制的話比較有用
  - cons: 
    - 同kernel2 
    - 如果分配的不好則是會更慢

assumption:如果記憶體的資料沒有很大的話會是k2比較快，但如果記憶體的資料很大則是k1比較快，而經過HW2的經驗kernel3會是最慢的
## Q2
>How are the performances of the three methods? Plot a chart to show the differences among the three methods


| view1 | view2 |
| -------- | -------- |
| ![](https://i.imgur.com/MdVzCTm.png)  |![](https://i.imgur.com/AksUIW8.png)  |

## Q3
>Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? 

在kernel3部分，最慢是有的，因為在切工作的時候再view1可以很明顯地知道，他們工作量大的是集中在一起的因此給每一個thread更多的pixel去做處理，只會造成原本很慢的thread會更慢而已。
至於kernel1以及kernel2的部分是在於Allco的部分1會比2還要快。
## Q4
>Can we do even better? Think a better approach and explain it. 


因為一開始在img就已經有宣告空間了因此我使用了ZERO COPY的技術來改善。
![](https://i.imgur.com/Q26pgVD.png)
成果 :
| view1 | view2 |
| -------- | -------- |
|![](https://i.imgur.com/UBep1gy.png)|![](https://i.imgur.com/D1uTstR.png)|



參考資料
--
[CUDA的4種記憶體存取方式](https://kaibaoom.tw/2020/07/21/cuda-four-memory-access/)
[zeor copy](https://www.twblogs.net/a/5b8ee2032b717718834876f2)
[cudaMallocPitch](https://www.twblogs.net/a/5cd92d5dbd9eee6726c9ece8)
