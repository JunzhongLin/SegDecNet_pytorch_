import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--loss_func', choices=['MSE', 'cross_entropy', 'dice'], help='MSE/cross_entropy/dice')


args = parser.parse_args()
print(vars(args))

if __name__=='__main__':
    if args.loss_func == 'MSE':
        print('Done')
