public int[] twoSum(int[] numbers, int target) {
        // //目标值减去第一个值，然后找有没有匹配的值
        int[] numbers2 = new int[2];
        int tar = 0;
        for(int i=0;i<numbers.length;i++){
            tar = target-numbers[i];
            for(int x = i+1;x<numbers.length;x++){
                //System.out.print(x,i);
                if(tar == numbers[x]){
                    numbers2[0] = i+1;
                    numbers2[1] = x+1;
                    break;
                }
            }
            if(numbers[0]==i+1)
            {
                break;
            }
        }
        return numbers2;
    }

int[] test = new [1,2,3,4,4,9,56,90];
x= twoSum(test,8);
