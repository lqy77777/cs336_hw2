# 如何连接 AutoDL 并配置 Codex

本文介绍如何从本地电脑通过 SSH 连接 AutoDL 远程服务器、配置公钥免密登录、把本地网络代理安全地转发给远程服务器，并安装 Codex CLI、Codex IDE 扩展和 Codex CLI 插件。

本文以以下网络结构为例：

```text
AutoDL 上的程序
    -> 远程 127.0.0.1:17897
    -> SSH RemoteForward 隧道
    -> 本地 127.0.0.1:7890
    -> 本地代理软件
    -> Internet
```

请先按自己的实际情况替换下列占位符：

| 占位符 | 含义 | 示例 |
| --- | --- | --- |
| `<AUTODL_HOST>` | AutoDL 控制台显示的 SSH 主机 | `connect.example.autodl.com` |
| `<AUTODL_PORT>` | AutoDL 控制台显示的 SSH 端口 | `12345` |
| `<AUTODL_USER>` | SSH 用户名 | 通常为 `root` |
| `<LOCAL_PROXY_PORT>` | 本地代理软件提供的 HTTP/Mixed 端口 | 本文假设为 `7890` |
| `17897` | 在 AutoDL 回环地址上监听的转发端口 | 可保留本文取值 |

> 重要：远程服务器中的 `127.0.0.1` 指 AutoDL 自己，不是你的本地电脑。只有配置 `RemoteForward` 后，AutoDL 的 `127.0.0.1:17897` 才能通过 SSH 隧道连接本地代理。

---

## 1. 修改本地电脑的 SSH config

### 1.1 找到 SSH 配置文件

Linux 和 macOS：

```text
~/.ssh/config
```

Windows：

```text
C:\Users\你的用户名\.ssh\config
```

如果 `.ssh` 目录或 `config` 不存在，可以创建它们。

Linux/macOS：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\.ssh"
New-Item -ItemType File -Force "$HOME\.ssh\config"
```

### 1.2 基本 SSH 配置

将以下内容写入本地 `.ssh/config`：

```sshconfig
Host autodl
    HostName <AUTODL_HOST>
    User <AUTODL_USER>
    Port <AUTODL_PORT>

    IdentityFile ~/.ssh/id_ed25519_autodl
    IdentitiesOnly yes

    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes

    ExitOnForwardFailure yes
    RemoteForward 127.0.0.1:17897 127.0.0.1:<LOCAL_PROXY_PORT>
```

例如，本地代理端口为 `7890` 时：

```sshconfig
Host autodl
    HostName connect.example.autodl.com
    User root
    Port 12345

    IdentityFile ~/.ssh/id_ed25519_autodl
    IdentitiesOnly yes

    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes

    ExitOnForwardFailure yes
    RemoteForward 127.0.0.1:17897 127.0.0.1:7890
```

`RemoteForward` 的含义是：

```text
远程服务器 127.0.0.1:17897
-> SSH 隧道
-> 本地电脑 127.0.0.1:7890
```

`127.0.0.1:17897` 只绑定到 AutoDL 的回环接口，不会直接向公网暴露本地代理。

### 1.3 如果 SSH 连接本身也必须经过代理

`RemoteForward` 解决的是“连接成功后，远程程序如何使用本地代理”。如果从本地连接 AutoDL 的 SSH TCP 连接本身也必须经过代理，还需要添加 `ProxyCommand`。

#### 方案 A：本地代理提供 HTTP CONNECT

本地安装 Nmap 后可使用 `ncat`：

```sshconfig
    ProxyCommand ncat --proxy 127.0.0.1:<LOCAL_PROXY_PORT> --proxy-type http %h %p
```

完整示例：

```sshconfig
Host autodl
    HostName connect.example.autodl.com
    User root
    Port 12345
    IdentityFile ~/.ssh/id_ed25519_autodl
    IdentitiesOnly yes

    ProxyCommand ncat --proxy 127.0.0.1:7890 --proxy-type http %h %p

    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    ExitOnForwardFailure yes
    RemoteForward 127.0.0.1:17897 127.0.0.1:7890
```

#### 方案 B：本地代理提供 SOCKS5

使用 `ncat`：

```sshconfig
    ProxyCommand ncat --proxy 127.0.0.1:<LOCAL_PROXY_PORT> --proxy-type socks5 %h %p
```

Linux/macOS 使用 OpenBSD `nc` 时，也可以写成：

```sshconfig
    ProxyCommand nc -x 127.0.0.1:<LOCAL_PROXY_PORT> -X 5 %h %p
```

HTTP 和 SOCKS5 方案只保留一个，不要同时写两个 `ProxyCommand`。

如果不经过代理也能连接 AutoDL，则不需要 `ProxyCommand`，但仍可保留 `RemoteForward`，让远程服务器使用本地网络代理。

### 1.4 检查 SSH config

查看展开后的配置：

```bash
ssh -G autodl
```

测试连接并显示详细日志：

```bash
ssh -v autodl
```

若 `RemoteForward` 建立失败，`ExitOnForwardFailure yes` 会阻止 SSH 悄悄建立一个没有代理隧道的会话。

---

## 2. 配置 SSH 公钥免密登录

### 2.1 在本地生成专用密钥

先检查是否已经存在：

```bash
ls -l ~/.ssh/id_ed25519_autodl ~/.ssh/id_ed25519_autodl.pub
```

如果不存在，在本地电脑执行：

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_autodl -C "autodl"
```

会生成：

```text
~/.ssh/id_ed25519_autodl       私钥，不能发给任何人
~/.ssh/id_ed25519_autodl.pub   公钥，可以放到服务器
```

推荐给私钥设置 passphrase。若不想每次输入 passphrase，可以在本地使用 `ssh-agent`，而不是创建完全无保护的私钥。

查看本地公钥：

```bash
cat ~/.ssh/id_ed25519_autodl.pub
```

Windows PowerShell：

```powershell
Get-Content "$HOME\.ssh\id_ed25519_autodl.pub"
```

### 2.2 将公钥写入 AutoDL

第一次仍使用 AutoDL 密码登录。Linux/macOS 如果有 `ssh-copy-id`：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_autodl.pub autodl
```

如果没有 `ssh-copy-id`，可以使用：

```bash
cat ~/.ssh/id_ed25519_autodl.pub | ssh autodl \
  'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
```

PowerShell：

```powershell
Get-Content "$HOME\.ssh\id_ed25519_autodl.pub" |
    ssh autodl "umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys"
```

然后登录服务器，检查权限：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 2.3 验证免密登录

退出后重新连接：

```bash
ssh autodl
```

如果私钥设置了 passphrase，系统可能要求输入的是本地私钥 passphrase，而不是 AutoDL 登录密码。这仍属于公钥认证。

不要关闭当前仍能登录的 SSH 窗口，直到确认新窗口可以正常使用密钥登录，以免错误配置导致自己被锁在服务器外。

---

## 3. 在 AutoDL 安装基本系统组件

以下命令均在 AutoDL 服务器内执行。

AutoDL 通常使用 `root` 用户。如果当前不是 root，请在 `apt-get` 前添加 `sudo`。

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    unzip \
    zip \
    jq \
    ripgrep \
    tmux \
    htop \
    tree \
    openssh-client \
    netcat-openbsd
```

更新证书：

```bash
update-ca-certificates
```

检查工具：

```bash
git --version
curl --version
rg --version
tmux -V
```

如果项目需要 Python 环境，可根据项目要求另外安装 Miniconda、uv 或系统 Python。不要随意覆盖 AutoDL 镜像自带的 CUDA、NVIDIA driver 或 PyTorch，除非已经确认版本兼容关系。

---

## 4. 让 AutoDL 使用本地网络代理

本节假设 SSH config 中已经存在：

```sshconfig
RemoteForward 127.0.0.1:17897 127.0.0.1:<LOCAL_PROXY_PORT>
```

只要这条 SSH 会话保持连接，AutoDL 就可以把 `127.0.0.1:17897` 当作 HTTP 代理地址。

### 4.1 先验证 SSH 代理隧道

在 AutoDL 中检查端口：

```bash
ss -lnt | grep 17897
```

直接通过代理测试：

```bash
curl -I --proxy http://127.0.0.1:17897 https://chatgpt.com
```

也可以检查出口 IP：

```bash
curl --proxy http://127.0.0.1:17897 https://api.ipify.org
echo
```

如果这里失败，先解决 SSH 隧道，不要继续配置 Codex。

### 4.2 写入 VS Code Remote Settings JSON

使用 VS Code Remote-SSH 连接 AutoDL 后，打开命令面板：

```text
Preferences: Open Remote Settings (JSON)
```

注意必须打开 Remote Settings，而不是本地 User Settings。

写入以下完整配置：

```json
{
    "remote.autoForwardPorts": false,
    "http.proxy": "http://127.0.0.1:17897",
    "http.proxySupport": "on",
    "terminal.integrated.env.linux": {
        "HTTP_PROXY": "http://127.0.0.1:17897",
        "HTTPS_PROXY": "http://127.0.0.1:17897",
        "http_proxy": "http://127.0.0.1:17897",
        "https_proxy": "http://127.0.0.1:17897",
        "ALL_PROXY": "http://127.0.0.1:17897",
        "all_proxy": "http://127.0.0.1:17897",
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1"
    }
}
```

这是合法 JSON，没有末尾多余逗号。VS Code 的 JSONC 通常允许注释和尾逗号，但保持标准 JSON 更容易复用和排查错误。

`remote.autoForwardPorts: false` 只关闭 VS Code 自动检测并转发远程端口，不会关闭 SSH config 中显式配置的 `RemoteForward`。

修改后执行：

```text
Developer: Reload Window
```

然后新建一个 VS Code integrated terminal，检查：

```bash
env | grep -i proxy
```

### 4.3 写入 AutoDL 的持久 shell 环境变量

仅配置 `terminal.integrated.env.linux` 只保证新建的 VS Code integrated terminal 获得这些变量。普通 SSH shell、`tmux` 和其他进程未必继承，所以还应配置远程 shell 环境。

创建独立代理环境文件：

```bash
mkdir -p ~/.config
nano ~/.config/proxy.env
```

写入：

```bash
export HTTP_PROXY="http://127.0.0.1:17897"
export HTTPS_PROXY="http://127.0.0.1:17897"
export http_proxy="http://127.0.0.1:17897"
export https_proxy="http://127.0.0.1:17897"
export ALL_PROXY="http://127.0.0.1:17897"
export all_proxy="http://127.0.0.1:17897"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
```

在 `~/.bashrc` 末尾加入：

```bash
if [ -f "$HOME/.config/proxy.env" ]; then
    . "$HOME/.config/proxy.env"
fi
```

如果使用 zsh，则写入 `~/.zshrc`。

立即加载：

```bash
source ~/.bashrc
```

验证：

```bash
env | grep -i proxy
curl -I https://chatgpt.com
```

Codex CLI 会继承启动它的 shell 环境，因此这些变量也是 Codex 的上游 HTTP/HTTPS 代理配置。

### 4.4 可选：给常用工具单独配置代理

如果某些工具不读取通用环境变量，可以单独配置。

Git：

```bash
git config --global http.proxy http://127.0.0.1:17897
git config --global https.proxy http://127.0.0.1:17897
```

npm：

```bash
npm config set proxy http://127.0.0.1:17897
npm config set https-proxy http://127.0.0.1:17897
```

pip：

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

内容：

```ini
[global]
proxy = http://127.0.0.1:17897
```

这些单独配置会在通用环境变量被取消后继续生效。若以后不再使用代理，需要分别清理。

撤销 Git 配置：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

撤销 npm 配置：

```bash
npm config delete proxy
npm config delete https-proxy
```

---

## 5. 安装和配置 Codex CLI

根据当前官方 OpenAI 文档，macOS/Linux 推荐使用 standalone installer：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

如果希望先检查脚本再执行：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/install-codex.sh
less /tmp/install-codex.sh
sh /tmp/install-codex.sh
```

安装完成后重新加载 shell：

```bash
source ~/.bashrc
```

检查：

```bash
command -v codex
codex --version
```

进入项目目录并首次启动：

```bash
cd /path/to/your/project
codex
```

首次运行时，根据终端提示选择可用的登录方式。远程终端若显示需要在浏览器中打开的地址，应在本地浏览器完成授权，然后返回终端继续。

官方 Codex CLI 文档：[Codex CLI](https://learn.chatgpt.com/docs/codex/cli)。

### 5.1 `config.toml` 的位置

用户级配置文件的官方位置是：

```text
~/.codex/config.toml
```

首次运行 Codex 通常会初始化 `~/.codex` 的相关状态，但不能假设所有版本都一定自动创建一个非空 `config.toml`。应先检查：

```bash
ls -la ~/.codex
test -f ~/.codex/config.toml && sed -n '1,200p' ~/.codex/config.toml
```

如果文件不存在，可以安全创建：

```bash
mkdir -p ~/.codex
touch ~/.codex/config.toml
chmod 600 ~/.codex/config.toml
```

官方配置参考：[Codex Configuration Reference](https://learn.chatgpt.com/docs/config-file/config-reference)。

### 5.2 关于在 `config.toml` 中加入 `[proxy]`

当前官方 Codex `config.toml` reference 中没有通用的顶层 `[proxy]` 配置，也没有用 `[proxy]` 设置 Codex 上游 HTTP 代理的标准写法。因此不要盲目加入：

```toml
[proxy]
http = "http://127.0.0.1:17897"
https = "http://127.0.0.1:17897"
```

未知配置可能被忽略，也可能在未来版本中导致校验错误。

当前可靠做法是让 Codex 进程继承：

```text
HTTP_PROXY
HTTPS_PROXY
ALL_PROXY
```

也就是前面已经写入 `~/.config/proxy.env` 和 VS Code Remote Settings 的环境变量。

官方文档中的 `[features.network_proxy]` 是 Codex sandboxed networking 的内部网络策略和监听器设置，不等同于“把 Codex 请求发送到上游 HTTP 代理”，不应为了普通代理需求直接照搬。

如果某个特定 Codex fork、公司内部版本或未来版本明确要求 `[proxy]`，应以该版本的 `codex --help`、release notes 和配置 reference 为准，再加入对应字段。

### 5.3 验证 Codex 是否继承代理

在启动 Codex 的同一个 shell 中检查：

```bash
env | grep -i proxy
curl -I https://chatgpt.com
codex --version
```

然后运行：

```bash
codex
```

进入 Codex 后可以使用：

```text
/status
```

查看当前 session configuration。

---

## 6. 安装 Codex 扩展和 Codex CLI 插件

“Codex 插件”可能指两种不同的东西：

1. VS Code 中的 Codex IDE extension；
2. Codex CLI marketplace 中的 plugin。

二者需要分别安装。

### 6.1 安装 VS Code Codex IDE extension

在本地 VS Code 中打开 Extensions，搜索官方 Codex 扩展并安装。使用 Remote-SSH 连接 AutoDL 后，如果 VS Code 显示 `Install in SSH: autodl`，再点击一次，将需要在远程 extension host 运行的部分安装到 AutoDL 环境。

安装或启用后：

1. 打开 AutoDL 项目目录；
2. 点击侧边栏 Codex 图标；
3. 若图标不可见，打开 Command Palette；
4. 执行 `Codex: Open Codex Sidebar`；
5. 按提示完成登录。

官方文档：[Codex IDE extension](https://learn.chatgpt.com/docs/codex/ide)。

### 6.2 安装 Codex CLI marketplace 插件

官方文档说明，Codex CLI 使用内置 plugin browser。先启动：

```bash
codex
```

然后输入：

```text
/plugins
```

在 plugin browser 中：

1. 切换 marketplace tab；
2. 搜索目标 plugin；
3. 打开详情并检查它包含的 skill、connector、MCP server 或 hook；
4. 选择 Install；
5. 如需外部服务，按提示完成认证；
6. 安装后退出当前 Codex session；
7. 重新启动一个新的 Codex CLI session。

新 session 才会加载刚安装 plugin 提供的 skills 和 tools。

官方文档：[Plugins](https://learn.chatgpt.com/docs/plugins)。

> 注意：官方文档当前说明 Codex IDE extension 不支持 CLI plugin browser。安装和管理 marketplace plugins 应使用 Codex CLI 的 `/plugins`，而不是在 IDE extension 中寻找该命令。

### 6.3 安装插件前的安全检查

Plugin 可能包含：

- skills；
- connectors；
- MCP servers；
- hooks；
- 外部认证流程。

安装前应检查来源和权限，尤其要查看 hook 会执行什么命令、connector 会访问什么数据。不要向来源不明的 plugin 提供 AutoDL 密钥、SSH 私钥或云平台凭证。

---

## 7. 保险操作：创建目录并修复权限

在 AutoDL 中执行：

```bash
mkdir -p ~/.codex
mkdir -p ~/.cache
mkdir -p ~/.config
mkdir -p /tmp
chmod 700 ~/.codex
chmod 700 ~/.cache ~/.config
chmod 1777 /tmp
```

含义：

- `~/.codex`：Codex 用户配置、认证和状态目录；
- `~/.cache`：用户级缓存目录；
- `~/.config`：用户级配置目录；
- `/tmp`：系统临时目录；
- `700`：只有当前用户可以读、写和进入目录；
- `1777`：所有用户都可在 `/tmp` 创建文件，但 sticky bit 阻止普通用户删除其他用户的文件。

还建议保护 Codex 配置文件：

```bash
test -f ~/.codex/config.toml && chmod 600 ~/.codex/config.toml
```

需要注意：`chmod 1777 /tmp` 是系统级操作，只有 root 能执行。AutoDL 单用户容器通常可以执行；在多人共享机器上，应先确认自己有管理权限，并且 `/tmp` 就是系统公共临时目录。

---

## 8. 完整验收清单

### SSH

- [ ] `ssh -G autodl` 展开配置正确；
- [ ] `ssh autodl` 可以连接；
- [ ] 新窗口可以通过公钥认证登录；
- [ ] 私钥权限安全，且没有上传到服务器；
- [ ] SSH 断开后能通过 keepalive 正常发现并重连。

### 代理

- [ ] 本地代理正在监听 `<LOCAL_PROXY_PORT>`；
- [ ] SSH config 包含 `RemoteForward 127.0.0.1:17897 ...`；
- [ ] AutoDL 中 `ss -lnt | grep 17897` 能看到监听；
- [ ] AutoDL 中显式 `curl --proxy` 可以访问外网；
- [ ] 新 SSH shell 中 `env | grep -i proxy` 显示完整变量；
- [ ] 新 VS Code remote terminal 中也能看到代理变量。

### Codex

- [ ] `command -v codex` 有输出；
- [ ] `codex --version` 正常；
- [ ] `~/.codex/config.toml` 存在且权限为 `600`；
- [ ] Codex 能完成登录并启动 session；
- [ ] `/status` 能显示当前 session 状态；
- [ ] VS Code Codex sidebar 可以打开；
- [ ] CLI 中 `/plugins` 可以打开 plugin browser；
- [ ] 安装 plugin 后已经重新启动 Codex session。

---

## 9. 常见故障排查

### 9.1 AutoDL 的 `127.0.0.1:17897` 拒绝连接

可能原因：

- SSH config 没有 `RemoteForward`；
- 修改 config 后没有重建 SSH 连接；
- 本地代理软件没有运行；
- `<LOCAL_PROXY_PORT>` 写错；
- AutoDL 已有其他进程占用 17897；
- SSH server 禁止 TCP forwarding。

检查：

```bash
ss -lntp | grep 17897
```

本地使用详细日志重连：

```bash
ssh -vvv autodl
```

### 9.2 `remote port forwarding failed for listen port 17897`

远程端口通常已被旧 SSH 会话占用。先关闭旧连接，或在 AutoDL 检查占用者：

```bash
ss -lntp | grep 17897
```

也可以更换远程端口，但必须同步修改：

- SSH `RemoteForward`；
- VS Code `http.proxy`；
- `terminal.integrated.env.linux`；
- `~/.config/proxy.env`；
- Git/npm/pip 的独立代理配置。

### 9.3 普通 SSH 终端能联网，但 VS Code/Codex 不能

执行：

```bash
env | grep -i proxy
```

确认 Codex 是从已经加载 `~/.bashrc` 的新 terminal 启动。修改 Remote Settings 后执行 `Developer: Reload Window`，修改 shell 配置后关闭并重新创建 terminal。

### 9.4 VS Code 能联网，但 `apt-get` 不能

`terminal.integrated.env.linux` 只影响 integrated terminal 启动的进程。若使用 `sudo`，默认可能清理代理环境变量。可尝试：

```bash
sudo -E apt-get update
```

但不要无条件允许所有环境变量穿过 sudo。在 AutoDL 的 root shell 中通常不需要 sudo。

### 9.5 Codex 安装成功但命令不存在

检查：

```bash
find "$HOME" -maxdepth 3 -type f -name codex 2>/dev/null
echo "$PATH"
```

按照 installer 输出提示把安装目录加入 `PATH`，然后：

```bash
source ~/.bashrc
hash -r
```

### 9.6 加入 `[proxy]` 后 Codex 报配置错误

删除未被当前官方配置 reference 支持的 `[proxy]` table，保留 shell 环境变量：

```bash
env | grep -i proxy
```

再检查：

```bash
codex --version
codex
```

---

## 10. 最短执行顺序

```text
1. 本地修改 ~/.ssh/config
2. 本地生成 id_ed25519_autodl
3. 把公钥追加到 AutoDL ~/.ssh/authorized_keys
4. 使用 ssh autodl 验证免密登录
5. 在 AutoDL 安装 curl、git、build-essential、rg、tmux 等组件
6. 用 SSH RemoteForward 建立 AutoDL:17897 -> 本地代理端口
7. 在 VS Code Remote Settings 写入完整 proxy JSON
8. 在 AutoDL ~/.config/proxy.env 与 ~/.bashrc 写入代理变量
9. 用 curl 验证代理
10. 安装并首次运行 Codex CLI
11. 检查 ~/.codex/config.toml，不加入官方未支持的通用 [proxy]
12. 安装 VS Code Codex extension
13. 在 Codex CLI 中通过 /plugins 安装所需插件
14. 重启 Codex session
15. 创建 ~/.codex、~/.cache、~/.config、/tmp 并修复权限
16. 按验收清单逐项检查
```

---

## 11. 官方参考

- [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
- [Codex Configuration Reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Codex IDE extension](https://learn.chatgpt.com/docs/codex/ide)
- [ChatGPT and Codex Plugins](https://learn.chatgpt.com/docs/plugins)

Codex 的安装方式、配置键和 plugin 支持范围可能随版本变化。遇到本文与当前 CLI 行为不一致时，应优先查看上述官方文档和本机 `codex --help`。
