# Media-and-cognition-experiment
an experiment of the course media and cognition-Traffic sign recognition

考完试闲得没事干，随便学学github，顺手记录一下媒认的图像识别实验

# introduction
虽然是自己做的作业，但其实就是各种博客代码的缝合怪版本，实验平台是python，主要用到了pytorch，scikitlearn，opencv之类的包  
实验的内容主要是交通标志的分类与检测

## 传统学习方法
这里用了hog+svm，其实也没怎么经过自己思考（知识匮乏以至于没有思考空间），看着助教的推荐就选了。  
hog用的opencv里的，svm用的sklearn里的，都比较基础。

## fasterrcnn做交通标志检测
基本就是复制粘贴pytorch官网的教程，dataset需要自己读一下形式之后自己用课程的训练集做一个  
另外感叹一下最艰难的还是配环境的过程，之后就是一些小修小补外加写写测试用的json文件了。
