print("Starting Torch imports...")
import torch
import torch.nn as nn
from torch.optim import Rprop
print("Torch Imports good. Starting cv2 imports...")
import cv2
import time
import random
import pyaudio
from playsound import playsound
import numpy as np
print("Imports good. Defining model...")
CHUNK = 8192  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sample rate (samples per second)
sound = {0:"b",1:"d",2:"f",3:"g",4:"h",5:"j",6:"k",7:"l",8:"m",9:"n",10:"ng",11:"p",12:"r",13:"s",14:"t",15:"v",16:"w",17:"y",18:"z",19:"zh",20:"ch",21:"sh",22:"th(u)",23:"th(v)"}
vowels = {0:"a",1:"e",2:"i",3:"o",4:"u",5:"oo",6:"ai",7:"ee",8:"ie",9:"oa",10:"ui",11:"yu",12:"oi",13:"ow",14:"er",15:"air",16:"ar",17:"or",18:"aw",19:"ear",20:"ure"}
torch.autograd.set_detect_anomaly(True)

sensory_outputs = 7

class visual_sensory(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(30000+25+2, sensory_outputs)
        self.activation = nn.Sigmoid()
        
 
    def forward(self, x):
        x = self.activation(self.hidden(x))
        return x
visual_sensory_model = visual_sensory()
visual_sensory_optimizer = Rprop(visual_sensory_model.parameters(), lr=0.5)
print(visual_sensory_model)
print("Visual model good.")
class audio_sensory(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4096+25+2, sensory_outputs)
        self.activation = nn.Sigmoid()
        
 
    def forward(self, x):
        x = self.activation(self.hidden(x))
        return x

audio_sensory_model = audio_sensory()
audio_sensory_optimizer = Rprop(audio_sensory_model.parameters(), lr=0.5)
print(audio_sensory_model)
print("Audio model good.")
class heuristic(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(98, 7)
        self.activation = nn.LeakyReLU()
        
 
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        return x
heuristic_model = heuristic()
heuristic_optimizer = Rprop(heuristic_model.parameters(), lr=0.000001)
print("Heuristic model good.")

class intentionality(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(49, 7)
        self.activation = nn.LeakyReLU()
        self.hidden2 = nn.Linear(7, 2)
        self.output = nn.Softmax(dim=0)
 
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x
intentionality_model = intentionality()
intentionality_optimizer = Rprop(intentionality_model.parameters(), lr=0.0000001)
criterion_i = nn.CrossEntropyLoss()
print("Intentionality model good.")

class transduction_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(49, 7)
        self.activation = nn.LeakyReLU()
        self.hidden2 = nn.Linear(7, 25)
        self.output = nn.Softmax(dim=0)
 
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x
transduction_c_model = transduction_c()
transduction_c_optimizer = Rprop(transduction_c_model.parameters(), lr=0.00000001)
criterion_tc = nn.CrossEntropyLoss()

class transduction_v(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(49, 7)
        self.activation = nn.LeakyReLU()
        self.hidden2 = nn.Linear(7, 22)
        self.output = nn.Softmax(dim=0)
 
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x
transduction_v_model = transduction_v()
transduction_v_optimizer = Rprop(transduction_v_model.parameters(), lr=0.00000001)
criterion_tv = nn.CrossEntropyLoss()
print("Transduction model good. Starting preview...")

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

randomDist = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6]
STmems_v = []
STmems_a = []
hconcat = []
for i in range(7):
    STmems_v.append(torch.zeros(7))
    STmems_a.append(torch.zeros(7))
Hmems = {}
intentioncounter_n = 0
intentioncounter_y = 0
spoke = True
vowel = False
last = torch.zeros(25+2)
while rval:
    #start = time.time()
    frame = cv2.resize(frame, (100, 100))
    data = stream.read(CHUNK)
    numpy_data = np.frombuffer(data, dtype=np.int16)
    fft_result = np.fft.fft(numpy_data)
    fft_result = np.abs(fft_result)[:CHUNK // 2]
    a = visual_sensory_model(torch.cat((last, torch.mul(torch.tensor(frame.flatten(), dtype=torch.float32), (1.0/255)))))
    b = audio_sensory_model(torch.cat((last, torch.tensor(fft_result, dtype=torch.float32))))
    randomnum = random.choice(randomDist)
    '''
    if(randomnum <= 1):
        visual_sensory_loss = torch.sum(torch.square(torch.sub(a, STmems_v[0], alpha=1)))
        visual_sensory_loss.backward(retain_graph=True)
        visual_sensory_optimizer.step()
        visual_sensory_optimizer.zero_grad()
        audio_sensory_loss = torch.sum(torch.square(torch.sub(b, STmems_a[0], alpha=1)))
        audio_sensory_loss.backward(retain_graph=True)
        audio_sensory_optimizer.step()
        audio_sensory_optimizer.zero_grad()
    '''
    if(not spoke):
        if(random.random() < 0.5):
            STmems_v[randomnum] = a
            STmems_a[randomnum] = b
    else:
        if(random.random() < 0.1):
            STmems_v[randomnum] = a
            STmems_a[randomnum] = b
    hout = heuristic_model(torch.cat(((torch.cat((torch.cat(random.sample(STmems_v, 6)),a))), torch.cat((torch.cat(random.sample(STmems_a, 6)),b)))))
    keynum = round(torch.sum(hout).item()/10)
    if(keynum in Hmems):
        '''
        if(len(Hmems[keynum])==7):
            if(randomnum == 0):
                heuristic_loss = torch.sum(torch.square(torch.sub(hout, Hmems[keynum][randomnum], alpha=1)))
                heuristic_loss.backward(retain_graph=True)
                heuristic_optimizer.step()
                heuristic_optimizer.zero_grad()
        '''
        if(random.random() < 0.05):
            Hmems[keynum].append(hout.clone().detach())
        if(random.random() < 0.05 and len(Hmems[keynum]) > 1):
            Hmems[keynum].pop(0)
    else:
        Hmems[keynum] = [hout.clone().detach()]
    if(len(Hmems[keynum]) >= 7):
        hconcat = random.sample(Hmems[keynum], 7)
    else:
        hconcat = Hmems[keynum].copy()
        n = 0
        while(True):
            n = (n * -1)-1
            if((keynum+n) in Hmems):
                hconcat.append(Hmems[keynum+n][0])
                break
            n = n * -1
            if((keynum+n) in Hmems):
                hconcat.append(Hmems[keynum+n][0])
                break
            if(n>10):
                break
        for i in range(7 - len(hconcat)):
            hconcat.append(torch.zeros(7))
    final_input = torch.cat(hconcat)
    tcout = transduction_c_model(final_input)
    tvout = transduction_v_model(final_input)
    intention_raw = intentionality_model(final_input)
    intention = torch.argmax(intention_raw).item()
    if(random.random()<0.1):
        if(len(Hmems[keynum]) > 10):
            intentionality_loss = criterion_i(intention_raw, torch.tensor([0.55,0.45], dtype=torch.float32))
            intentionality_loss.backward(retain_graph=True)
            intentionality_optimizer.step()
            intentionality_optimizer.zero_grad()
            Hmems[keynum].pop(0)
        else:
            intentionality_loss = criterion_i(intention_raw, torch.tensor([0.6,0.4], dtype=torch.float32))
            intentionality_loss.backward(retain_graph=True)
            intentionality_optimizer.step()
            intentionality_optimizer.zero_grad()
    '''
    if(intention == 0):
        intentioncounter_n = intentioncounter_n + 1
    else:
        intentioncounter_y = intentioncounter_y + 1
    if(intentioncounter_n > 20):
        intentioncounter_n = 0
        transduction_loss = criterion_t(tout, torch.zeros(45))
        transduction_loss.backward(retain_graph=True)
        transduction_optimizer.step()
        transduction_optimizer.zero_grad()
        intentionality_loss = criterion_i(intention_raw, torch.tensor([0.5,0.5], dtype=torch.float32))
        intentionality_loss.backward(retain_graph=True)
        intentionality_optimizer.step()
        intentionality_optimizer.zero_grad()
    elif(intentioncounter_y > 3):
        intentioncounter_y = 0
        intentionality_loss = criterion_i(intention_raw, torch.tensor([0.05,0.95], dtype=torch.float32))
        intentionality_loss.backward(retain_graph=True)
        intentionality_optimizer.step()
        intentionality_optimizer.zero_grad()
        '''
    
    sindex = torch.argmax(tcout).item()
    vindex = torch.argmax(tvout).item()
    print(intention_raw)
    if(intention != 0):
        if(vowel):
            if(vindex != 21):
                print(vowels[vindex])
                last = torch.cat((tvout.clone().detach(), torch.zeros(3), intention_raw.clone().detach()))
                playsound(vowels[vindex] + ".mp3", False)
        else:
            if(sindex != 24):
                print(sound[sindex])
                last = torch.cat((tcout.clone().detach(), intention_raw.clone().detach()))
                playsound(sound[sindex] + ".mp3", False)
        vowel = not vowel
        spoke = True
    else:
        print("Didn't speak")
        vowel = not vowel
        spoke = False

    #end = time.time()
    #print("Time: " + str(end-start))


    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    #if key == 27: # exit on ESC
    #    break
    #time.sleep(10)
    print("-----------------------------")

vc.release()
cv2.destroyWindow("preview")