import numpy
#used sigmoid func 
import scipy.special
import matplotlib.pyplot
from time import time
# read of img
import imageio.v2 as imageio
 
start = time() 
class NeuralNetwork:
    def __init__(self):
        
        self.inodes = 784
        self.hnodes = 200
        self.onodes = 10
        # w_i_j (neuron i aas neuron j ruu holboson)w11 w21
        #w12 w22 etc, biasuud n togtmol - 0.5
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #learning rate
        self.lr = 0.1

        #activation function with sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
    
    # study function
    def train(self, inputs_list,targets_list):
        # oroltuud 2d array ruuu hurvuulne
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #signaluudaa dald davharg ruu totsoolh
        hidden_inputs = numpy.dot(self.wih, inputs)
        #used activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        #signaluuda garltiin davhargt totsoolh
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # singaluuda garltiin davhraga deer idevhjvvlegch func totsoo heseg
        final_outputs = self.activation_function(final_inputs)
        # alda tootsoh (target - actual)
        output_errors = targets - final_outputs
        #dald davharg deerh aldanuud
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #Bakprop ashiglan garaltiin davhargaas dald davharg ruu aldaa zasah heseg
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        #bakprop ashiglan dald davhargaas oroltiin davharg ruu aldaa zasah
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # surgsan hiiimal oyunaas asuult asuuh func
    def query(self, inputs_list):
        # oroluud 2d massiiv ruu horvuulnee
        inputs = numpy.array(inputs_list, ndmin=2).T
        # signal dald davharg ruu tootsoo heseg
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        #signaluud garalt davhargr totsoo heseg
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #signaluudaa grarltiin davhraga deer idevhejvvlh func deer totsoo hiih
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# oroltiin, dald, garaltiin neuronuud
# input_nodes = 784
# hidden_nodes = 200
# output_nodes = 10
# # learning rate
# learning_rate = 0.1
# Create Neuron network
print("started")
n = NeuralNetwork()
# Learning MNIST run data
training_data_file = open("./mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#Neuron svljee surgah 
#Epochs, surgaltiin vi
epochs = 10
for e in range(epochs):
    # Surgaltiin data deer davtalt hiih
    print(f"training ... {round(time() - start,2)} seconds")
    for record in training_data_list:
        
        all_values = record.split(',')
        #Normalchlah
        inputs = (numpy.asfarray(all_values[1:]) / 255.0*0.99) + 0.01
        targets = numpy.zeros(10)+0.01
        targets[int(all_values[0]) ]=0.99
        n.train(inputs,targets)
        pass
    print("epoch----",e)
    pass

# sursan neuron svljeeg test hiih
img_array = imageio.imread('./0000.png',mode='L')
#28x28 aas 784 utagtai
img_data = img_array.reshape(784)
print("---]] ",img_data)

#Normalchilh
img_data = (img_data / 255.0 * 0.99)+0.01
print("min =", numpy.min(img_data))
print("max =", numpy.max(img_data))

#end
matplotlib.pyplot.imshow(img_data.reshape(28,28),cmap='Greys', interpolation='None')
#surgsan neuronoos asuuh shalgah
outputs = n.query(img_data)
print(outputs)
#Hamgiin ih magdlaltai utaga
label = numpy.argmax(outputs)
print("Hiimel oyunii hariu",label,"<<<<<<<<<<<")
print(f"finished after {round(time() - start,2)} seconds")```
```
# Welcome to my profile 🔆 🌩️
<!-- Redis rabbitmq -->
#### I'm Khuzair - 21 years old Junior developer 🕵️‍♀️

* 🌐 <a herf="https://www.figma.com/file/7exZOtR4OToYZ461fX3JUI/Untitled?type=design&node-id=6-2&mode=design&t=U6Eicjr1TjKvP1bQ-0">Visit my website here!</a>
* 🎓 I'm myself a university 😄
* 👨🏻‍💻 dev @huzarh
*  KZ/MN
* 🌍 Based in Turkey, Mongolia and Kazakhstan
* 🌄 Painter artist
* 🚀🎨 Interested in all things space & art

#### doing it now 💻http://turk--ce.com/
---

## 🔬Technologies and skills I use



 
[![My Skills](https://skillicons.dev/icons?i=react,js,ts,next,vue,threejs,redux,mui,nodejs,mongodb,vite,regex,powershell,nginx,nextjs,mysql,linux,github,git,firebase,figma,express,emotion,firebase)](https://www.instagram.com/zir_huz/)

## ✨ Drew

<div id="header" align="center" style="display:flex;">
   
  <img src="https://scontent.fkco5-1.fna.fbcdn.net/v/t39.30808-6/166982124_344879250607103_6803994875227046076_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=19026a&_nc_ohc=4ktuG_ckIzEAX_K8QNf&_nc_ht=scontent.fkco5-1.fna&oh=00_AfC3n7wFAiUxAjlry1OlQWlydyQhX2E4m6TQSwGSmepjhQ&oe=643981C4" width="400" hieght="150" style="border-radius:10px"/>
   &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://scontent.fkco5-1.fna.fbcdn.net/v/t39.30808-6/332266543_529606719313721_6590504967901402580_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=730e14&_nc_ohc=YIlvDMDnvaQAX_5nNs_&_nc_ht=scontent.fkco5-1.fna&oh=00_AfB_BmgPvoUgZRTo9uIl60fpCDfGK_gopGB7SQtrjFg-MA&oe=64392FFA" width="155" style="border-radius:10%" borderRadius="20%"  />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://scontent.fkco5-1.fna.fbcdn.net/v/t1.6435-9/130975824_214000263695003_8572418138708170922_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=p09NQJmK_vwAX8IIfbE&_nc_ht=scontent.fkco5-1.fna&oh=00_AfAx7ahGl2Dnzkb5KnEVtRFN4VI7MYMj0fpe_wxbBBYBpw&oe=645B71B7" width="146" style="border-radius:10%" borderRadius="20%"  /> 
</div>

---
## ⏰ Free time:

<div id="badges">
  <a href="your-linkedin-URL">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="your-youtube-URL">
    <img src="https://img.shields.io/badge/YouTube-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Youtube Badge"/>
  </a>
  <a href="your-twitter-URL">
    <img src="https://img.shields.io/badge/Twitter-blue?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter Badge"/>
  </a>
   <a href="your-facebook-URL">
    <img src="https://img.shields.io/badge/Facebook-blue?style=for-the-badge&logo=facebook&logoColor=white" alt="Facebook Badge"/>
  </a>
   <a href="">
    <img src="https://img.shields.io/badge/ChatGPT-brightgreen?style=for-the-badge&logo=chatgpt&logoColor=white" alt="ChatGPT Badge"/>
   </a>
   <a href="">
    <img src="https://img.shields.io/badge/sport-brightgreen?style=for-the-badge&logo=sport&logoColor=white" alt="ChatGPT Badge"/>
   </a>
  
</div>

![nature](https://i.pinimg.com/originals/f9/47/74/f94774094cdb0632c80e94a27d4de239.gif)

---

### :fire: My Stats :

[![GitHub Streak](http://github-readme-streak-stats.herokuapp.com?user=huzarh&theme=dark&background=000000)](https://git.io/streak-stats,https://github.com/anuraghazra/github-readme-stats)

<!-- [![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=huzarh)](https://github.com/anuraghazra/github-readme-stats) -->

<!--- requir: 
real time laptop public ip url: ngrokf
npm key:0ec36ea883f13acc6edcba717085afb09e24874de5c2f39e831f1730897ef523
4a7b9a977f51a67ef9b4c2caf1bb0578bf6408e24acb531b6064ad93a1a1e22b
44db7155a75d9780551ffb401b833b684dae9d20ceaacf81c0a170c844992205
5c86e3b984313e02746b4b017a2f81ea13479b406cbd46c81694f6eec9930957
5790de671ec78c2a6d6764a30e9104971a0305e05472ed3198ec41285d9d787a


huzarh/huzarh is a ✨ special ✨ repository because sdv its `README.md` (this file) appears on your GitHub profile.
You can click the Pr.  ergseg er g.   reg egrr.  eview  ewr gw eg  look at your changes ewr gw eg  look at your changes link to take adsvs ewr gw eg  look at your changes.
  ergseg er g.   reg egrr.  eview  ewr gw eg  look at your changes ewr gw eg  look at your changes link to take adsvs ewr gw eg  look at your changes.ink to take adsvs ewr gw eg  look at your changes.


odev

1. bisection method
2. baker's donusumu
3. mandelbrot fraktal
--->
