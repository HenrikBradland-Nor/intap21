from config import *
import config


def system_init():
    SUPORTS_CUDA = torch.cuda.is_available()
    if (SUPORTS_CUDA):
        print("Cuda availeble, current device: cuda:0")
        return "cuda:0"
    else:
        print("Cuda not availeble, current device: cpu")
        return "cpu"


def showPC(pc, colur, rgb=True):
    pc = pc.detach().numpy()
    pc -= np.mean(pc, 0)
    c = colur.detach().numpy()
    #c -= c.min() # Scaling values to 0-255
    #c /= c.ptp()
    #c *= 255
    #c = np.asarray(c, dtype=np.uint8)

    pc_rgb = pcl.PointCloud.PointXYZRGB()
    pc_mono = pcl.PointCloud.PointXYZ()

    pc_rgb = pc_rgb.from_array(pc*0.01, c)
    pc_mono = pc_mono.from_array(pc*0.01)
    visual = pclpy.pcl.visualization.CloudViewer('test window')
    if rgb:
        visual.showCloud(pc_rgb, 'test')
    else:
        visual.showCloud(pc_mono, 'test')
    v = True
    while v:
        v = not (visual.wasStopped())

class timer:
    def __init__(self):
        self.start = time()
        self.last = self.start
        self.log = []

    def resetTimer(self):
        self.last = time()

    def timeStamp(self, text=None):
        if text is None:
            self.log.append(['Event #' + str(len(self.log)+1), time() - self.last])
        else:
            self.log.append([text, time() - self.last])
        self.last = time()

    def printLog(self):
        print("=" * 30)
        for e in self.log:
            print(e[0], "%.3f" % e[1], '[sec]')
        print("-" * 30)
        print("Total time", "%.3f" % (time() - self.start), "[sec]")
        print("=" * 30 + "\n\n")

    def emptyLog(self):
        self.log = []
        self.resetTimer()

    def getLog(self):
        return self.log
