import matplotlib.pyplot as plt


def plot(spectrum):
    """
    spectrum: 2d numpy array
    """
    for j in range(int(len(spectrum)/16+1)):
        fig = plt.figure(figsize=(16, 8))
        #plt.title('label: star; predict: qso')
        #plt.title('label: star; predict: galaxy')
        #plt.title('label: qso; predict: galaxy')
        #plt.title('label: qso; predict: star')
        #plt.title('label: galaxy; predict: star')
        # plt.title('label: galaxy; predict: qso')
        plt.axis('off')
        for k in range(16):
            if j*16+k >= len(spectrum):
                break
            ax = fig.add_subplot(4, 4, k+1)
            plt.plot(spectrum[j*16+k], 'b-', linewidth=0.5)
        plt.show()
