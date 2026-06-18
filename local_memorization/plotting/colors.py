import matplotlib

def dimcolors(dim,): 
    colors =['#a0b2ff', '#a2b2fd', '#a4b2fc', '#a6b2fa', '#a8b2f9', '#abb3f7', '#adb3f6', '#afb3f4', '#b0b3f2', '#b2b3f1', '#b4b3ef', '#b6b3ee', '#b8b3ec', '#bab3eb', '#bbb4e9', '#bdb4e7', '#bfb4e6', '#c0b4e4', '#c2b4e3', '#c4b4e1', '#c5b4e0', '#c7b4de', '#c8b4dd', '#cab5db', '#ccb5d9', '#cdb5d8', '#cfb5d6', '#d0b5d5', '#d1b5d3', '#d3b5d2', '#d4b5d0', '#d6b6ce', '#d7b6cd', '#d8b6cb', '#dab6ca', '#dbb6c8', '#dcb6c7', '#deb6c5', '#dfb7c3', '#e0b7c2', '#e1b7c0', '#e3b7bf', '#e4b7bd', '#e5b7bc', '#e6b7ba', '#e7b8b8', '#e9b8b7', '#eab8b5', '#ebb8b4', '#ecb8b2', '#edb8b0', '#eeb8af', '#efb9ad', '#f1b9ac', '#f2b9aa', '#f3b9a8', '#f4b9a7', '#f5b9a5', '#f6b9a4', '#f7baa2', '#f8baa0', '#f9ba9f', '#faba9d']
    num_colors = 50 
    color_step = int(num_colors/dim)
    return colors[::-color_step]

def time_colors(num_ts,): 
    colors = ['#596900', '#5b6b09', '#5e6d11',
        '#606f18', '#63711e', '#657323',
        '#687528', '#6a772d', '#6c7832',
        '#6e7a37', '#717c3b', '#737e40',
        '#758045', '#778249', '#7a844e',
        '#7c8652', '#7e8857', '#808a5b',
        '#828c60', '#848e64', '#869069',
        '#88926e', '#8a9572', '#8c9777',
        '#8e997b', '#909b80', '#929d84',
        '#949f89', '#96a18e', '#98a392',
        '#9aa597', '#9ba79c', '#9da9a0',
        '#9faba5', '#a1aeaa', '#a3b0af',
        '#a4b2b3', '#a6b4b8', '#a8b6bd',
        '#a9b8c2', '#abbac7', '#adbdcc',
        '#aebfd0', '#b0c1d5', '#b1c3da',
        '#b3c5df', '#b4c7e4', '#b6cae9',
        '#b8ccee', '#b9cef3']
    num_colors = 50 
    color_step = int(num_colors/num_ts)
    return colors[::-color_step]


def loss_color(key): 
    color_list = ["#bdabae","#2274a5",
                  "#222725","#8e3b46",
                  "#90b494"]
    colors = {"loss" : color_list[0],
              "loss_theo": color_list[1], 
              "loss_opt" : color_list[2], 
              "loss_diag": color_list[3],
              "test_loss": color_list[4]}
    return colors[key]
    
def spike_colors(num_values):
    colors = ['#005c66', '#15626c', '#236773',
              '#2e6c79', '#387280', '#417786',
              '#4a7c8d', '#538294', '#5c879b',
              '#658da1', '#6d92a8', '#7698af',
              '#7e9db6', '#87a3bd', '#90a8c4',
              '#98aecc', '#a1b4d3', '#a9b9da',
              '#b2bfe1', '#bbc5e9', '#c3caf0',
              '#ccd0f8', '#fff9dd', '#fff2da',
              '#ffebd7', '#ffe5d3', '#ffded0',
              '#ffd8cc', '#ffd1c8', '#ffcac3',
              '#fec4bf', '#fcbeba', '#fbb8b5',
              '#f9b2b0', '#f7acab', '#f4a6a5',
              '#f1a19e', '#ee9c98', '#ea9790',
              '#e59288', '#e08f7f', '#da8b75',
              '#d28969', '#c88859']
    num_colors = len(colors)
    color_step = int(num_colors/num_values)
    return colors[::-color_step]