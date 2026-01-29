#include <iostream>
using namespace std;

class box{
    public:
    int val;
    box operator+(const box &b){
        box temp;
        temp.val=this->val + b.val;
        return temp;
    }

};

int main(){
    box a,b,d;
    a.val=10;
    b.val=20;
    d.val=30;
    box c=a+b+d; // a.operator+(b);
    cout<<c.val;
    return 0;
}