import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        # Sử dụng log_softmax thay vì softmax để tránh vấn đề số học
        return torch.nn.functional.log_softmax(logits, dim=-1)

class Policy2210xxx(Policy):
    def __init__(self):
        self.state_dim = 200
        self.action_dim = 100
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Khởi tạo device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def _encode_state(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Chuẩn hóa features
        stocks_features = []
        total_space = 0
        used_space = 0
        
        for stock in stocks:
            stock_array = np.array(stock)
            available = (stock_array == -1).sum()
            used = (stock_array >= 0).sum()
            total_space += available + used
            used_space += used
            stocks_features.extend([available/100.0, used/100.0])  # Chuẩn hóa giá trị
            
        # Thêm tỷ lệ sử dụng tổng thể
        stocks_features.append(used_space/max(1, total_space))
        
        # Chuẩn hóa thông tin sản phẩm
        products_features = []
        total_demand = 0
        for prod in products:
            size = prod["size"]
            quantity = prod["quantity"]
            total_demand += quantity * size[0] * size[1]
            products_features.extend([size[0]/10.0, size[1]/10.0, quantity/10.0])  # Chuẩn hóa giá trị
            
        # Padding features
        stocks_features = np.array(stocks_features, dtype=np.float32)
        products_features = np.array(products_features, dtype=np.float32)
        
        stocks_features = np.pad(stocks_features, 
                               (0, max(0, 100 - len(stocks_features))), 
                               mode='constant')[:100]
        
        products_features = np.pad(products_features, 
                                 (0, max(0, 100 - len(products_features))), 
                                 mode='constant')[:100]
        
        state = np.concatenate([stocks_features, products_features])
        return torch.FloatTensor(state).to(self.device)

    def get_action(self, observation, info):
        state = self._encode_state(observation)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                log_probs = self.policy_network(state)
                probs = torch.exp(log_probs)
                action_idx = torch.multinomial(probs, 1).item()
        
        # Lưu log probability cho training
        log_prob = self.policy_network(state)[action_idx]
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action_idx)
        
        return self._decode_action(action_idx, observation)

    def update_policy(self):
        if len(self.rewards) == 0:
            return
            
        # Tính returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        
        # Chuẩn hóa returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Tính loss và cập nhật policy
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)  # Thêm gradient clipping
        self.optimizer.step()
        
        # Giảm epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Reset memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def _decode_action(self, action_idx, observation):
        # Giữ nguyên phần decode_action như cũ
        stocks = observation["stocks"]
        products = observation["products"]
        
        valid_products = [(i, p) for i, p in enumerate(products) if p["quantity"] > 0]
        if not valid_products:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        prod_idx, prod = valid_products[action_idx % len(valid_products)]
        prod_size = prod["size"]
        
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size
            
            if stock_w < prod_w or stock_h < prod_h:
                continue
                
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        return {
                            "stock_idx": stock_idx,
                            "size": prod_size,
                            "position": (x, y)
                        }
        
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
