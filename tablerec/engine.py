import torch
from datetime import datetime


def info(message):
    cur = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    print(f'{cur} {message}', flush=True)


class LossRecord:
    def __init__(self, loss = 0., count = 0):
        self.loss = loss
        self.count = count

    def __add__(self, item):
        self.loss += item
        self.count += 1
        return self

    def mean(self):
        if not self.count:
            return 0.
        return self.loss / self.count

    def zero(self):
        self.count = 0
        self.loss = 0.


def run_epoch(data, model, device, criterion, optimizer, epoch):
    loss = {
        'row': LossRecord(),
        'col': LossRecord(),
        'running': LossRecord(),
        'total': LossRecord()
    }
    for index, batch in enumerate(data):
        if model.training:
            optimizer.zero_grad()

        image = batch['image'].to(device)
        row_split_pos = batch['row_split'].to(device)
        col_split_pos = batch['col_split'].to(device)

        row_pred, col_pred = model(image)
        row_loss = criterion(row_pred, row_split_pos)
        col_loss = criterion(col_pred, col_split_pos)
        cur_loss = row_loss + col_loss

        loss['total'] += cur_loss
        loss['running'] += cur_loss
        loss['row'] += row_loss
        loss['col'] += col_loss

        if model.training:
            cur_loss.backward()
            optimizer.step()

        if device == 0:
            template = '{key}({value:.3f})'
            message = ' '.join([
                template.format(key=x, value=loss[x].mean()) for x in loss
            ])
            info('[{i}/{n}, {e}]{msg}'.format(
                i=index+1, n=len(data), e=epoch+1, msg=message
            ))
            loss['running'].zero()
            loss['row'].zero()
            loss['col'].zero()

    if device == 0:
        info(f'Epoch {epoch+1} total loss: {loss["total"].mean():.3f}')


def train_epoch(data, model, device, criterion, optimizer, epoch):
    model.train()
    return run_epoch(data, model, device, criterion, optimizer, epoch)


def val_epoch(data, model, device, criterion, optimizer, epoch):
    model.eval()
    with torch.no_grad():
        return run_epoch(data, model, device, criterion, optimizer, epoch)
