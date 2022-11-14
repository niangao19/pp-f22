# 平行程式作業-hw2
###### tags:  `HW` `PP`
## Q1
>Q1: In your write-up, produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1. 

使用環境:MACOS M2
| thread num| 1 | 2 | 3 | 4 |
| -------- | -------- | -------- | -------- | -------- |
| view1-time   (ms)  |   381.72	| 201.214|	239.524|	165.76|
| view1-speed up|  1	|1.89|	1.59|2.3|
|view2-time (ms) |201.188| 126.291|	100.256|	85.838|
|view2-speed up | 1|	1.61|	2.02|	2.37|





| 1.thread cost time | 2.speed up |
| -------- | -------- |
| ![](https://i.imgur.com/PoeSmwQ.png) |![](https://i.imgur.com/N6p783C.png)

:::info
> Is speedup linear in the number of threads used?
* 可以根據表2的圖可以看到當thread num上升時，在view 1 時speed up並不會線性上升，主要原因是因為每一個thread處理的資料複雜度並不是完全相同(ex:part1之題目就是所有thread處理皆相同數目)，還有在建立thread時會需要時間來建立，因此速度並不會直線上升。
:::


| view 1 | view 2 | 
| -------- | -------- | 
| ![](https://i.imgur.com/a21UEhT.png)  | ![](https://i.imgur.com/fxDNbQ4.jpg)|

``` cpp
float dx = (x1 - x0) / width;
float dy = (y1 - y0) / height;
//使用for迴圈算出x y值後輸入到mandel function
float x = x0 + i * dx;
float y = y0 + j * dy;
// c_re = x , c_im = y
int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}
``` 
上方程式碼為 MandelbrotSerial.cpp 中的 mandel() 函數，透過此函數可以看到每個像素被賦值的過程，並從中得出以下幾點：
* 像素的count是由輸入的maxIterations來決定,並賦予給for跑的次數。
* 並且在for迴圈跑的次數越多，i的數值就越大，在圖片中呈現出的顏色就越淺。
:::info
>In your writeup hypothesize why this is (or is not) the case? (You may also wish to produce a graph for VIEW 2 to help you come up with a good answer. Hint: take a careful look at the three-thread data-point.)
* 可以觀察上面表格兩個圖片可以發現此兩張圖差別最多的部分是關於顏色的區塊差異，並且在thread num 為3時，可以觀察到view 1的speed up 下降了許多，可以推測出有些thread執行的運算數較大，也就是說分配到產出的圖片較多白色的部分，因此導致了整體執行速度不佳。



:::



## Q2
>Q2: How do your measurements explain the speedup graph you previously created?

在Q1中可以觀察到在view 1、2中差別最大的部分就是白色部分大小。
在原先的程式中，我們是以切塊來區別每一條Thread要處理的部分。
``` cpp
int numRows = data->height/numThreads;
int startRow = numRows * threadId;
mandelbrotSerial(
        data->x0, data->y0, data->x1, data->y1,
        data->width, data->height,
        startRow, numRows, data->maxIterations, data->output);
```


| thread num | 1 | 2 |3 | 4|
| -------- | -------- | -------- | -------- | -------- |
| view 1     |  ![](https://i.imgur.com/X6wqQnj.png)| ![](https://i.imgur.com/Dw1tWAi.png)|![](https://i.imgur.com/wOIl624.png)|![](https://i.imgur.com/WVrb706.png)
| view 2 | ![](https://i.imgur.com/TsjCIiB.jpg)|![](https://i.imgur.com/vXOjkzu.jpg)|![](https://i.imgur.com/G33D7Lx.jpg)|![](https://i.imgur.com/lGWDduk.jpg)







thread num 1 : 1
```shell
./mandelbrot -t 1
The 0 thread : cost 0.381293 time

./mandelbrot -t 1 --view 2
The 0 thread : cost 0.203833 time
```

thread num 2:
```shell
./mandelbrot -t 2
The 0 thread : cost 0.400385 time
The 1 thread : cost 0.403519 time

./mandelbrot -t 2 --view 2
The 1 thread : cost 0.165754 time
The 0 thread : cost 0.211483 time
```
可以觀察到在view1中，所處理的資料量是差不多的，因此所執行的時間也相近。
但在view2中上面那塊明顯有比較多白色像素，因此ThreadId0所執行的時間較長。

thread num 3 :
```shell
./mandelbrot -t 3
The 0 thread : cost 0.233021 time
The 2 thread : cost 0.240690 time
The 1 thread : cost 0.404632 time

./mandelbrot -t 3 --view 2
The 2 thread : cost 0.166337 time
The 1 thread : cost 0.173001 time
The 0 thread : cost 0.217574 time
```
view 1 : 分成三段中見thread id 1分配比較多的白色部分，因此花費的時間比另外兩條多。
view 2 : thread id 0分配很多白色部分，花得比較多時間。

thread num 4 :
```shell
./mandelbrot -t 4
The 0 thread : cost 0.150389 time
The 3 thread : cost 0.153555 time
The 1 thread : cost 0.404325 time
The 2 thread : cost 0.408950 time

./mandelbrot -t 4 --view 2
The 3 thread : cost 0.159522 time
The 1 thread : cost 0.165038 time
The 2 thread : cost 0.175695 time
The 0 thread : cost 0.217943 time
```
view 1 : 分成三段中見thread id 1、2分配比較多的白色部分，因此花費的時間比另外兩條多。
view 2 : thread id 0分配很多白色部分，花得比較多時間。
:::info
結論 : 因為每一條的thread分配的工作量不同，因此speed up無法線性增加。
:::



## Q3
>Q3: In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained

以先前的題目可以知道，無法讓thread num上升時，speed up無法直線性增加是因為，每一條thread的工作量分配不均。
* 不已分區的方式分配給thread工作，改以一行一行分配。
``` cpp
for(unsigned int j = threadId; j < data->height; j += numThreads  ) {
        for (unsigned  int i = 0; i < data->width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;
            int index = (j * data->width + i);
            output[index] = mandel(x, y, data->maxIterations);
        } //for
} // for
```
執行時間
```shell
% ./mandelbrot -t 4
[mandelbrot serial]:		[679.063] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:		[170.632] ms
Wrote image file mandelbrot-thread.ppm
				(3.98x speedup from 4 threads)
```

:::info
view 1 (3.98x speedup from 4 threads)
:::
## Q4
>Q4: Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not?

使用環境：i5 4cores
```shell
$ ./mandelbrot -t 4
[mandelbrot serial]:            [459.640] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [120.895] ms
Wrote image file mandelbrot-thread.ppm
                                (3.80x speedup from 4 threads)
                                
$ ./mandelbrot -t 8
[mandelbrot serial]:            [458.989] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [124.763] ms
Wrote image file mandelbrot-thread.ppm
                                (3.68x speedup from 8 threads)
```

:::info
可以看到thread num為4時，他的speed up為3.80X，而thread num為8時，他的speed up為3.68X。
明顯的發現當thread num成長到8時speed up並無明顯的上升，主要是因為這次使用的環境是在i5 4核心的狀況下去做實驗，此sever一次只能平行跑4條thread，因此超過此thread量後，就不是平行去執行行的，而是4條4條thread concurrent去執行，反而會造成一些額外的context switch之成本，所以thread num往上升並不會跑得更快。
:::







