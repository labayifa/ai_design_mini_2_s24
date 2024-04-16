##  Mini 2 Lab 3 csagbo
We implemented a dashboard to :
- Add a batch execution test function

### Docker config
> We Create Dockerfile config to build and push our image to docker hub

> This project files are described below:
> - __*app_iris.py*__ : The Rest API written in python with Flask
> - __*base_iris_lab1.py*__ : 
> - __*iris_extended_encoded.csv*__ :
> - __*request_client.py*__ : The client driver
> - __*logs*__ : Folder containing the logs listed below

> -![Docker Image pushed](./logs/docker_push_image_logs.png)
> -![Docker Ran_Container](./logs/run_container_logs.jpeg)
> -![Container_executed](./logs/run_exec_logs.jpeg)

> Also known as a Representational State Transfer Application programming interface, a Rest API is an architectural style for building distributed systems based on hypermedia around resources. An example of resource is for instance a customer managed by our systems of a machine learning model. When we are designing them some best practices should be followed: 
	- The type of request aligned with some HTTP Verbs:
		* GET: retrieve resource: GET /customers/1
		* GET : retrieve all resources : GET /customers
		* POST: creating a new resource:  POST /customers
		* PUT: create or replace an existing resource: PUT /customers
		* DELETE: to delete a resource: DELETE /customers/1
		* PATCH: update partially a resource

	The PUT, and POST, PATCH, DELETE requests can carry some request body. For GET requests we only need a request parameter (Query parameters or path variables), also applicable for the other HTTP Verbs. Apart from that the resources should be organized following the semantics above for best practices. 

> - After each request, there is a response with a given content and status code: ● 4xx: Error coming from client side
> ● 5xx: Server error
> ● 2xx: Successful request
> ● 3xx: Resource redirected.

> Other than all this every Rest resource should be accessed by specifying some Media Type specified by the developer in his documentation to accept incoming and sending the appropriate response (Application JSON, XML).
		