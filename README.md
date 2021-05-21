# image-triplet-representations

### Accessing the jupyter server on the dgx-1

1. If off-campus, connect to the Clemson VPN. For example, on linux, execute
    ```bash
    sudo openconnect cuvpn.clemson.edu
    ```
   and follow the prompts. The first password is your Clemson SSO password; 
   the second is your preferred 2-factor authentication option. I use "push"
   to receive a Duo push notification to my phone.
   The connection must stay open while developing.
2. In a new terminal, ssh into the development server with port forwarding:
    ```bash
    ssh -L 9240:localhost:9240 <clemson_username>@nvidiadgx1-01.clemson.edu
    ```
3. You can now access the jupyter lab environment in your browser. To get the access URL, execute
    ```
    docker exec image-triplet jupyter notebook list
    ```
   Paste the URL string output by the above command into your browser replacing the `0.0.0.0` with `localhost`. 
   You can save this URL and skip step 3 in the future. However, anytime the server is restarted, the access token 
   is reset and you will need to re-run step 3. So, if you find the URL is not working, re-run step 3 above
   and make sure that your saved URL matches the output.
