# Generated by Django 2.2.12 on 2022-01-18 13:43

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Classifier',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_image', models.ImageField(upload_to='')),
                ('output_image', models.ImageField(upload_to='')),
            ],
        ),
    ]
