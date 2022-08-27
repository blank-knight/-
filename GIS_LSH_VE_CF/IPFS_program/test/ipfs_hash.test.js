const { assert } = require('chai');
const { contracts_build_directory } = require('../truffle-config');

const ipfs_hash = artifacts.require("ipfs_hash");

require('chai')
    .use(require('chai-as-promised'))
    .should()

contract('ipfs_hash',(accounts) => {
    let ipfs
    
    before(async () => {
        ipfs = await ipfs_hash.deployed()
    })

    describe('deployment',async () => {
        it('deploys successfully', async () => {
            const address = ipfs.address
            console.log(address)
            assert.notEqual(address,0x0)
            assert.notEqual(address,'')
            assert.notEqual(address,null)
            assert.notEqual(address,undefined)
        })
    })

    describe('storage', async () => {
        it('updates the ipfs_hash', async () => {
            let ipfsTest
            ipfsTest = 'abc123'
            await ipfs.set(ipfsTest)
            const result = await ipfs.get() 
            assert.equal(result,ipfsTest)
        })
    })
})