
import torch
import numpy as np
from tqdm.auto import tqdm

'''
train 
test
'''

def train_loop(model, x, y, optimizer, loss_func, batch_size, hidden_state, clipping_theta, device):
    model.train()
    train_loss = 0
    train_X, train_Y = x, y
    iterations = len(train_Y) // batch_size
    # print(iterations) #1763
    for itr in range(iterations):
        train_x_data = torch.tensor(train_X[itr * batch_size:(itr + 1) * batch_size, :, :], dtype=torch.float).reshape(
            batch_size, train_X.shape[1], train_X.shape[-1]).to(device)
        train_y_data = torch.tensor(train_Y[itr * batch_size:(itr + 1) * batch_size], dtype=torch.float).to(device)
        if hidden_state is not None:
            # 使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
            if isinstance(hidden_state, tuple):  # LSTM, state:(h, c)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:
                hidden_state = hidden_state.detach()

        predicts_y, hidden_state = model(train_x_data, hidden_state)

        loss = loss_func(predicts_y, train_y_data)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_clipping(model.parameters(), clipping_theta, device)
    f_state = hidden_state
    train_loss = train_loss / iterations
    return train_loss, f_state


def test_loop(model, x, y, loss_func, batch_size, device, hidden_state):
    model.eval()

    test_loss = 0
    test_X, test_Y = x, y
    prediction_y = []

    iterations = len(test_Y) // batch_size
    for itr in range(iterations):
        test_x_data = torch.tensor(test_X[itr * batch_size:(itr + 1) * batch_size, :, :], dtype=torch.float).reshape(
            batch_size, test_X.shape[1], test_X.shape[-1]).to(device)
        test_y_data = torch.tensor(test_Y[itr * batch_size:(itr + 1) * batch_size], dtype=torch.float).to(device)
        predicts_y, _ = model(test_x_data, hidden_state)
        loss = loss_func(predicts_y, test_y_data)
        test_loss += loss.item()
        prediction_y.append(predicts_y.data.reshape(1).tolist())
    # print(np.array(prediction_y).shape)
    test_loss = test_loss / iterations
    return test_loss, np.array(prediction_y)


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train(model, train_X, train_Y, test_X, test_Y, optimizer, loss_func, batch_size, hidden_state, epochs, device,
          clipping_theta, lr, num_hiddens):
    model.to(device)
    results = {'train_loss': [], 'test_loss': []}
    for epoch in tqdm(range(epochs)):
        train_loss, f_state = train_loop(model=model, x=train_X, y=train_Y, optimizer=optimizer, loss_func=loss_func,
                                         batch_size=batch_size, hidden_state=hidden_state,
                                         clipping_theta=clipping_theta, device=device)
        test_loss, pred_y = test_loop(model=model, x=test_X, y=test_Y, loss_func=loss_func, batch_size=batch_size,
                                            device=device, hidden_state=f_state)

        print(f'Epoch:{epoch + 1}|', f'train_loss:{train_loss:.4f}', f'test_loss:{test_loss:.4f}')
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # torch.save(model.state_dict(),
    #            f'./od_pred/model_state/lstm_epoch{epochs}_numhidden{num_hiddens}_windowsize{train_X.shape[1]}_lr{lr}_train0.8.pth')

    return results, pred_y