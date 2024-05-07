import numpy
#used sigmoid func 
import scipy.special
import matplotlib.pyplot as plt
# read of img
import imageio.v2 as imageio 
plt.rcParams['figure.figsize']=(5.0,4.0)
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

        #signaluuda garltiin davhargt totsoolh ok
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # singaluuda garltiin davhraga deer idevhjvvlegch func totsoo heseg
        final_outputs = self.activation_function(final_inputs)
        # alda tootsoh (target - actual)
        output_errors = targets - final_outputs
        #dald davharg deerh aldanuud
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #Backprop ashiglan garaltiin davhargaas dald davharg ruu aldaa zasah heseg
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        #Backpropagation  ashiglan dald davhargaas oroltiin davharg ruu aldaa zasah
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
training_data_file = open("./test.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


#Neuron svljee surgah 
#Epochs, surgaltiin vi
epochs = 10
arg = 0
arg2 = 0
for e in range(epochs):
    # Surgaltiin data deer davtalt hiih
    for record in training_data_list:

        all_values = record.split(',')
        #Normalchlah
        inputs = (numpy.asfarray(all_values[1:]) / 255.0*0.99) + 0.01
        targets = numpy.zeros(10)+0.01
        targets[int(all_values[0]) ]=0.99 
        n.train(inputs,targets)
        pass
    print(f"wih & who: {len(n.wih)},{len(n.wih[0])} -- {len(n.who)},{len(n.who[0])}")
    print(f"\n{n.wih[0][:20]}\n------------------------\n{n.wih[0][:20]}")
    print("--------",len(n.who[0][:400]),len(n.wih[0][:400]))
    if(e==0):
     arg = n.wih[0][:20]
    elif(e==9):
     arg2= n.wih[0][:20]
    # if(e%2==0):
    #     plt.subplot(2, 2, 4)
    #     plt.scatter(n.who[0],n.wih[0][:200],cmap='coolwarm')
    #     plt.axis('off')
    #     plt.show()
    print("epoch----",e)

print(len(n.wih[0]),len(n.wih))
# numpy.save('wih.npy', n.wih)
# numpy.save('who.npy', n.who)

# Enlarge the page
# plt.figure(figsize=(20, 10))  # Adjust the width and height as needed
# plt.suptitle('args')
print(len(arg))
plt.scatter(numpy.arange(20), arg,color="blue") 
plt.scatter(numpy.arange(20), arg2,color="green") 
plt.show()
# plt.scatter(numpy.arange(len(arg2)-1), arg2) 
# plt.show()
# Iterate over your data
# for i in range(len(arg)):
#     print(f"--------------------------- 00 --------------------------\n{arg[i]}---{arg2[i]}\n==========================================================")

#     fig, (ax1, ax2) = plt.subplots(2,sharex=True)
#     fig.suptitle('Aligning x-axis using sharex')
#     ax1.scatter(arg[i], arg2[i])
# #     print(f"--------------ilk\n{arg[i]}\n{arg2[i]}\n----")
# #     # ax2.scatter(arg[i][0][:200], arg2[i][0][:200])
# #     # plt.subplot(10, 5, i + 1, frame_on=True)
# #     # plt.scatter(arg[i][0][:200], arg2[i][0][:200], cmap='coolwarm')
#     plt.axis('off')

# plt.show()
## --- | ------------------ | ----------------- | --- ##
    
plt.figure(figsize=(10, 5))
plt.suptitle('Weights from Input to Hidden Layer')
for i in range(n.hnodes):
    plt.subplot(10, 20, i + 1)
    plt.imshow(n.wih[i].reshape(28, 28), cmap='coolwarm', interpolation='nearest')  # Adjust visualization parameters
    plt.axis('off')
plt.show()

# sursan neuron svljeeg test hiih
img_array = imageio.imread('./Figure_2.png',mode='L')
#28x28 aas 784 utagtai
img_data = img_array.reshape(784)

#Normalchilh
img_data = (img_data / 255.0 * 0.99)+0.01

("min =", numpy.min(img_data))
print("max =", numpy.max(img_data))

#end
plt.imshow(img_data.reshape(28,28),cmap='Greys', interpolation='None')
#surgsan neuronoos asuuh shalgah
outputs = n.query(img_data)
#Hamgiin ih magdlaltai utaga
label = numpy.argmax(outputs)
print("Hiimel oyunii hariu",label,"<<<<<<<<<<<")
