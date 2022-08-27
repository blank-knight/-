const ipfs_hash = artifacts.require("ipfs_hash");

module.exports = function(deployer) {
  deployer.deploy(ipfs_hash);
};
