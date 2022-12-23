import threading
from multiprocessing import Pool


class myThread(threading.Thread):
    def __init__(self, func, objs, s, end, output=None):
        threading.Thread.__init__(self)
        self.func = func
        self.objs = objs
        self.s = s
        self.end = end
        self.output = output

    def run(self):
        for idx in range(self.s, self.end):
            if self.output != None:
                self.output.append(self.func(self.objs[idx]))
            else:
                self.objs[idx] = self.func(self.objs[idx])

    def getObjs(self):
        return self.objs


def runSerialLoop(func, array):
    for i in array:
        i = func(i)
    return array


def runThreadedLoop(func, array, output=None):
    j = len(array)
    threads = list()
    for i in range(5):
        threads.append(
            myThread(func, array, int(j // 6 * i), int(j // 6 * (i + 1)), output=output)
        )
    threads.append(myThread(func, array, int(j // 6 * 5), j, output=output))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return array


def runPoolMapLoop(func, array):
    with Pool(6) as p:
        ret = p.map(func, array)
    return ret


def runParallel(func, array, output=None):
    return runThreadedLoop(func, array, output)
