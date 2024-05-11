import restnet_1d_android as restnet_1d

def initial_train(train_loader, model, device, dtype, criterion, learning_rate):
    print("Initial training started...")
    restnet_1d.train(model, train_loader, device, dtype, criterion=criterion, learning_rate=learning_rate)
    print("Initial training complete!")

def gradient_train(traning_step, model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    print("Gradient trainining started at step: ", traning_step)
    gradients = restnet_1d.gradient_train(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion)
    print("Gradient training complete!")
    return gradients

def psuedo_labeling(model, devc, dtype, loader):
    print("Pseudo labeling started...")
    pseudo_labels = restnet_1d.psuedo_labeling(model, devc, dtype, loader)
    print("Pseudo labeling complete!")
    return pseudo_labels

    


