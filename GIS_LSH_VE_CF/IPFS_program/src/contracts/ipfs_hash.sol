// SPDX-License-Identifier: SimPL-2.0

pragma solidity ^0.8.13;

contract ipfs_hash{
    string memeHash;

    function set(string memory _memeHash) public {
        memeHash = _memeHash;
    }

    function get() public view returns (string memory){
        return memeHash;
    }
}