/*
SQLyog Community Edition- MySQL GUI v6.07
Host - 5.5.30 : Database - pills_detection
*********************************************************************
Server version : 5.5.30
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

create database if not exists `pills_detection`;

USE `pills_detection`;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

/*Table structure for table `pill_metadata` */

DROP TABLE IF EXISTS `pill_metadata`;

CREATE TABLE `pill_metadata` (
  `pill_name` varchar(100) DEFAULT NULL,
  `pill_usage` varchar(1000) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `pill_metadata` */

insert  into `pill_metadata`(`pill_name`,`pill_usage`) values ('pantoprazole 40 MG','Pantoprazole is a proton pump inhibitor (PPI) commonly used to reduce stomach acid production. It is prescribed for various conditions related to excess stomach acid, such as gastroesophageal reflux disease (GERD), peptic ulcers, and Zollinger-Ellison syndrome. '),('celecoxib 200 MG','Celecoxib is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, inflammation, and stiffness caused by certain conditions such as arthritis. '),('montelukast 10 MG','Montelukast is a medication commonly used to manage symptoms of asthma and allergic rhinitis. '),('duloxetine 30 MG','Duloxetine is a medication commonly used to treat major depressive disorder, generalized anxiety disorder, fibromyalgia, and certain types of chronic pain conditions. '),('Atomoxetine 25 MG','Atomoxetine is a medication commonly used to treat attention deficit hyperactivity disorder (ADHD).');

/*Table structure for table `users` */

DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
  `name` varchar(100) DEFAULT NULL,
  `userid` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `mobilenumber` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `users` */

insert  into `users`(`name`,`userid`,`password`,`email`,`mobilenumber`) values ('CLOUDTECHNOLOGIES','ct123','555555','ct@gmail.com','8121953811');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
