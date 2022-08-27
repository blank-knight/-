import React, { Component } from 'react';
import logo from '../logo.png';
import Web3 from 'web3';
import './App.css';
import ipfs_hash from '../abis/ipfs_hash.json'

// 调用ipfs-api，连接ipfs
const ipfsAPI = require('ipfs-api');
const ipfs = ipfsAPI({
host: 'localhost',
port: '5001',
protocol: 'http'
});

class App extends Component {

  // async componentWillMount(){
  //   await this.loadWeb3()
  //   await this.loadBlockchainData()
  // }

  // 加载区块链数据
  // async loadBlockchainData() {
  //   const web3 = window.web3 // 等待web3的发起
  //   const accounts = await web3.eth.getAccounts()
  //   this.setState({ account: accounts[0]})
  //   const networkId = await web3.eth.net.getId()
  //   console.log(accounts)
  //   console.log(networkId)
  //   const networkDate = ipfs_hash.networks[networkId] // 获取合约数据
  //   if(networkDate){
  //     const abi = ipfs_hash.abi
  //     const address = networkDate.address
  //     const contract = web3.eth.Contract(abi,address)
  //     this.setState({ contract })
  //     const ipfsHash = await contract.methods.get().call()
  //     this.setState({ ipfsHash })
  //     console.log(contract) // 输出合约信息
  //   }else{ // 如果当前网络没有获取到合约则输出
  //     window.alert('当前网络没有检测到部署的合约')
  //   }
  // }

  constructor(props){
    super(props);
    this.state = {
      account:'',
      contract:null,
      buffer:null,
      ipfsHash:'QmXVJLsasmYb13QzeLvApPCAqf8YCoydDtpN75qHFqDmZr'
    };
  }

  // async loadWeb3(){
  //   if(window.ethereum){
  //     window.web3 = new Web3(window.ethereum)
  //     await window.eth_requestAccounts
  //     // await window.ethereum.enable()
  //   }if(window.web3){
  //     window.web3 = new Web3(window.web3.currentProvider)
  //   }else{
  //     window.alert('Please use metamask!')
  //   }
  // }


  // captureFile = (event) => {
  //   event.preventDefault()
  //   const file = event.target.files[0]
  //   const reader = new window.FileReader()
  //   reader.readAsArrayBuffer(file)
  //   reader.onloadend = () => {
  //     this.setState({ buffer:Buffer(reader.result) })
  //   }
  //   console.log()
  // }

  // onSubmit = (event) => {
  //   event.preventDefault()
  //   console.log("提交表格")
  //   ipfs.add(this.state.buffer,(error, result) =>{ //向ipfs添加文件
  //     console.log('Ipfs result', result)
  //     const ipfsHash = result[0].hash
  //     console.log(ipfsHash)
  //     // this.setState({ipfs_hash})
  //     if(error){
  //       console.error(error)
  //       return
  //     }
  //     console.log('${this.state.ipfsHash}')
  //     // 存储进区块链
  //     // this.state.contract.methods.set(ipfsHash).send({ from: this.state.account}).then((r) => {
  //     //   this.setState({ipfsHash})
  //     // })
  //   })
  // }
  render() {
    return (
      <div>
        <nav className="navbar navbar-dark fixed-top bg-dark flex-md-nowrap p-0 shadow">
          <a
            className="navbar-brand col-sm-3 col-md-2 mr-0"
            href="http://www.dappuniversity.com/bootcamp"
            target="_blank"
            rel="noopener noreferrer"
          >
            去中心化传输网站
          </a>
          <ul className='navbar-nav px-3'>
            <li className='nav-item text-nownap d-none d-sm-block'>
              <small className='text-white'>
                {this.state.account}
              </small>
            </li>
          </ul>
        </nav>
        <div className="container-fluid mt-5">
          <div className="row">
            <main role="main" className="col-lg-12 d-flex text-center">
              <div className="content mr-auto ml-auto">
                <a
                  href="http://www.dappuniversity.com/bootcamp"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {/* <img src={logo} className="App-logo" alt="logo" /> */}
                  <img src={`https://ipfs.io/ipfs/${this.state.ipfsHash}`} alt="es-lint want to get"/> 
                </a>
                <p>&nbsp;</p>
                <h2>上传文件</h2>
                <form onSubmit={this.onSubmit}>
                  <input type='file' onChange={this.captureFile}/>
                  <input type='submit' />
                </form>
                {/* <h1>Dapp University Starter Kit</h1>
                <p>
                  Edit <code>src/components/App.js</code> and save to reload.
                </p>
                <a
                  className="App-link"
                  href="http://www.dappuniversity.com/bootcamp"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  LEARN BLOCKCHAIN <u><b>NOW! </b></u>
                </a> */}
              </div>
            </main>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
